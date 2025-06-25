import os
import numpy as np
import json
import pandas as pd
from calendar import monthrange
import torch
import utils
import netCDF4
from datetime import datetime, timedelta
from timecoding_modules import temporal_encoding

class LocationDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, input_enc, device):

        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster)

        # define some properties:
        self.locs = locs
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = labels
        #self.classes = classes
        #self.class_to_taxa = class_to_taxa

        # useful numbers:
        self.num_classes = len(np.unique(labels))
        self.input_dim = self.loc_feats.shape[1]

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        obs  = self.labels[index]
        return loc_feat, loc, obs
    
class TimeLocationDataset(torch.utils.data.Dataset):
    def __init__(self, locs, labels, input_enc, device, timestamps):

        # handle input encoding:
        self.input_enc = input_enc
        if self.input_enc in ['env', 'sin_cos_env']:
            raster = load_env()
        else:
            raster = None
        self.enc = utils.CoordEncoder(input_enc, raster)
        self.time_enc = temporal_encoding

        # define some properties:
        self.locs = locs
        self.timestamps = timestamps
        self.loc_feats = self.enc.encode(self.locs)
        self.labels = labels
        self.time_feats = self.time_enc(self.timestamps)
        #self.classes = classes
        #self.class_to_taxa = class_to_taxa

        # useful numbers:
        #self.num_classes = len(np.unique(labels))
        self.input_dim = self.loc_feats.shape[1]

        if self.enc.raster is not None:
            self.enc.raster = self.enc.raster.to(device)

    def __len__(self):
        return self.loc_feats.shape[0]

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc       = self.locs[index, :]
        obs  = self.labels[index]
        time = self.time_feats[index]
        return loc_feat, loc, obs, time
    

def load_env():
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    raster = load_context_feats(os.path.join(paths['env'],'bioclim_elevation_scaled.npy'))
    return raster

def load_context_feats(data_path):
    context_feats = np.load(data_path).astype(np.float32)
    context_feats = torch.from_numpy(context_feats)
    return context_feats

def load_inat_data(ip_file, taxa_of_interest=None):

    print('\nLoading  ' + ip_file)
    data = pd.read_csv(ip_file)

    # remove outliers
    num_obs = data.shape[0]
    data = data[((data['latitude'] <= 90) & (data['latitude'] >= -90) & (data['longitude'] <= 180) & (data['longitude'] >= -180) )]
    if (num_obs - data.shape[0]) > 0:
        print(num_obs - data.shape[0], 'items filtered due to invalid locations')

    if 'accuracy' in data.columns:
        data.drop(['accuracy'], axis=1, inplace=True)

    if 'positional_accuracy' in data.columns:
        data.drop(['positional_accuracy'], axis=1, inplace=True)

    if 'geoprivacy' in data.columns:
        data.drop(['geoprivacy'], axis=1, inplace=True)

    if 'observed_on' in data.columns:
        data.rename(columns = {'observed_on':'date'}, inplace=True)

    num_obs_orig = data.shape[0]
    data = data.dropna()
    size_diff = num_obs_orig - data.shape[0]
    if size_diff > 0:
        print(size_diff, 'observation(s) with a NaN entry out of' , num_obs_orig, 'removed')

    # keep only taxa of interest:
    if taxa_of_interest is not None:
        num_obs_orig = data.shape[0]
        data = data[data['taxon_id'].isin(taxa_of_interest)]
        print(num_obs_orig - data.shape[0], 'observation(s) out of' , num_obs_orig, 'from different taxa removed')

    print('Number of unique classes {}'.format(np.unique(data['taxon_id'].values).shape[0]))

    locs = np.vstack((data['longitude'].values, data['latitude'].values)).T.astype(np.float32)
    taxa = data['taxon_id'].values.astype(np.int64)

    if 'user_id' in data.columns:
        users = data['user_id'].values.astype(np.int64)
        _, users = np.unique(users, return_inverse=True)
    elif 'observer_id' in data.columns:
        users = data['observer_id'].values.astype(np.int64)
        _, users = np.unique(users, return_inverse=True)
    else:
        users = np.ones(taxa.shape[0], dtype=np.int64)*-1

    # Note - assumes that dates are in format YYYY-MM-DD
    years  = np.array([int(d_str[:4])   for d_str in data['date'].values])
    months = np.array([int(d_str[5:7])  for d_str in data['date'].values])
    days   = np.array([int(d_str[8:10]) for d_str in data['date'].values])
    days_per_month = np.cumsum([0] + [monthrange(2018, mm)[1] for mm in range(1, 12)])
    dates  = days_per_month[months-1] + days-1
    dates  = np.round((dates) / 364.0, 4).astype(np.float32)
    if 'id' in data.columns:
        obs_ids = data['id'].values
    elif 'observation_uuid' in data.columns:
        obs_ids = data['observation_uuid'].values

    return locs, taxa, users, dates, years, obs_ids

def find_all_nc_files(root_dir):
    nc_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.nc'):
                full_path = os.path.join(dirpath, file)
                nc_files.append(full_path)
    return nc_files

def load_obs_data(ip_dir):
    
    files = find_all_nc_files(ip_dir)

    def parse_nctime(s):
        return datetime.strptime(s, "%Y%m%d_%H%M%S")

    def generate_pixel_time_array(start, end, shape, axis='x'):
        total_secs = (end - start).total_seconds()
        H, W = shape
        if axis == 'x':
            lin_times = np.linspace(0, total_secs, W)  # 가로 scan
            scan_times = np.tile(lin_times, (H, 1)) + start.timestamp()
        else:
            lin_times = np.linspace(0, total_secs, H)  # 세로 scan
            scan_times = np.tile(lin_times.reshape(H, 1), (1, W)) + start.timestamp()
        return scan_times  # shape: (H, W)

    all_data = []
    
    for ip_file in files:
        print(f"\nLoading {ip_file}")
        
        ds = netCDF4.Dataset(ip_file)

        # 메타데이터 시간 파싱
        start_time = parse_nctime(ds.getncattr("observation_start_time"))
        end_time = parse_nctime(ds.getncattr("observation_end_time"))

        # 위경도 추출
        nav_group = ds.groups["navigation_data"]
        lat = nav_group.variables["latitude"][:]
        lon = nav_group.variables["longitude"][:]
        shape = lat.shape  # (H, W)
        
        geo = ds.groups["geophysical_data"]
        chl = geo.variables["Chl"][:]
        flag = geo.variables["flag"][:]

        # 시간 보간
        time_arr = generate_pixel_time_array(start_time, end_time, shape, axis='x')  # scan axis='x' assumed

        # 유효 범위 필터링
        valid_mask = (
            (lat >= -90) & (lat <= 90) &
            (lon >= -180) & (lon <= 180)
        )
        
        # 유효 픽셀만 추출
        lat_valid = lat[valid_mask]
        lon_valid = lon[valid_mask]
        chl_valid = chl[valid_mask]
        time_valid = time_arr[valid_mask]

        # 시간 → 연도, 정규화된 날짜
        timestamps = pd.to_datetime(time_valid, unit='s')
        years = timestamps.year.values.astype(np.int32)

        # 날짜 정규화 (월/일 기준 연속일 → 0~1 정규화)
        days = timestamps.day.values
        months = timestamps.month.values
        days_per_month = np.cumsum([0] + [31,28,31,30,31,30,31,31,30,31,30,31])
        julian = days_per_month[months - 1] + days
        dates = (julian / 365.0).astype(np.float32)

        # locs, obs, ids
        locs = np.vstack([lon_valid, lat_valid]).T.astype(np.float32)
        obs = chl_valid.astype(np.float32)
        obs_ids = np.arange(len(obs))

        all_data.append((locs, obs, dates, years, obs_ids))

    # 모든 파일에서 concat
    locs = np.concatenate([a[0] for a in all_data], axis=0)
    obs = np.concatenate([a[1] for a in all_data], axis=0)
    dates = np.concatenate([a[2] for a in all_data], axis=0)
    years = np.concatenate([a[3] for a in all_data], axis=0)
    obs_ids = np.concatenate([a[4] for a in all_data], axis=0)

    return locs, obs, dates, years, obs_ids


def load_timeseries_data(ip_dir):
    from timecoding_modules import temporal_encoding
    
    files = find_all_nc_files(ip_dir)

    def parse_nctime(s):
        return datetime.strptime(s, "%Y%m%d_%H%M%S")

    def generate_pixel_time_array(start, end, shape, axis='x'):
        total_secs = (end - start).total_seconds()
        H, W = shape
        if axis == 'x':
            lin_times = np.linspace(0, total_secs, W)
            scan_times = np.tile(lin_times, (H, 1)) + start.timestamp()
        else:
            lin_times = np.linspace(0, total_secs, H)
            scan_times = np.tile(lin_times.reshape(H, 1), (1, W)) + start.timestamp()
        return scan_times

    all_data = []

    for ip_file in files:
        print(f"\nLoading {ip_file}")
        ds = netCDF4.Dataset(ip_file)
        start_time = parse_nctime(ds.getncattr("observation_start_time"))
        end_time = parse_nctime(ds.getncattr("observation_end_time"))

        nav_group = ds.groups["navigation_data"]
        lat = nav_group.variables["latitude"][:]
        lon = nav_group.variables["longitude"][:]
        shape = lat.shape

        geo = ds.groups["geophysical_data"]
        chl = geo.variables["Chl"][:]
        flag = geo.variables["flag"][:]

        time_arr = generate_pixel_time_array(start_time, end_time, shape, axis='x')

        valid_mask = (
            (lat >= -90) & (lat <= 90) &
            (lon >= -180) & (lon <= 180)
        )

        lat_valid = lat[valid_mask]
        lon_valid = lon[valid_mask]
        chl_valid = chl[valid_mask]
        time_valid = time_arr[valid_mask]

        timestamps = pd.to_datetime(time_valid, unit='s')
        years = timestamps.year.values.astype(np.int32)

        days = timestamps.day.values
        months = timestamps.month.values
        days_per_month = np.cumsum([0] + [31,28,31,30,31,30,31,31,30,31,30,31])
        julian = days_per_month[months - 1] + days
        dates = (julian / 365.0).astype(np.float32)

        locs = np.vstack([lon_valid, lat_valid]).T.astype(np.float32)
        obs = chl_valid.astype(np.float32)
        obs_ids = np.arange(len(obs))

        all_data.append((locs, obs, dates, years, obs_ids, timestamps))

    locs = np.concatenate([a[0] for a in all_data], axis=0)
    obs = np.concatenate([a[1] for a in all_data], axis=0)
    dates = np.concatenate([a[2] for a in all_data], axis=0)
    years = np.concatenate([a[3] for a in all_data], axis=0)
    obs_ids = np.concatenate([a[4] for a in all_data], axis=0)
    timestamps = torch.cat([a[5] for a in all_data], dim=0)

    return locs, obs, dates, years, obs_ids, timestamps


def choose_aux_species(current_species, num_aux_species, aux_species_seed, taxa_file):
    if num_aux_species == 0:
        return []
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    taxa_file = os.path.join(data_dir, taxa_file)
    with open(taxa_file, 'r') as f:
        inat_large_metadata = json.load(f)
    aux_species_candidates = [x['taxon_id'] for x in inat_large_metadata]
    aux_species_candidates = np.setdiff1d(aux_species_candidates, current_species)
    print(f'choosing {num_aux_species} species to add from {len(aux_species_candidates)} candidates')
    rng = np.random.default_rng(aux_species_seed)
    idx_rand_aux_species = rng.permutation(len(aux_species_candidates))
    aux_species = list(aux_species_candidates[idx_rand_aux_species[:num_aux_species]])
    return aux_species

def get_taxa_of_interest(species_set='all', num_aux_species=0, aux_species_seed=123, taxa_file=None, taxa_file_snt=None):
    if species_set == 'all':
        return None
    if species_set == 'snt_birds':
        assert taxa_file_snt is not None
        with open(taxa_file_snt, 'r') as f: #
            taxa_subsets = json.load(f)
        taxa_of_interest = list(taxa_subsets['snt_birds'])
    else:
        raise NotImplementedError
    # optionally add some other species back in:
    aux_species = choose_aux_species(taxa_of_interest, num_aux_species, aux_species_seed, taxa_file)
    taxa_of_interest.extend(aux_species)
    return taxa_of_interest

def get_idx_subsample_observations(labels, hard_cap=-1, hard_cap_seed=123):
    if hard_cap == -1:
        return np.arange(len(labels))
    print(f'subsampling (up to) {hard_cap} per class for the training set')
    class_counts = {id: 0 for id in np.unique(labels)}
    ss_rng = np.random.default_rng(hard_cap_seed)
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in idx_rand:
        class_id = labels[i]
        if class_counts[class_id] < hard_cap:
            idx_ss.append(i)
            class_counts[class_id] += 1
    idx_ss = np.sort(idx_ss)
    print(f'final training set size: {len(idx_ss)}')
    return idx_ss

def get_train_data(params):
    with open('paths.json', 'r') as f:
        paths = json.load(f)
    data_dir = paths['train']
    #obs_file  = os.path.join(data_dir, params['obs_file'])
    #taxa_file = os.path.join(data_dir, params['taxa_file'])
    #taxa_file_snt = os.path.join(data_dir, 'taxa_subsets.json')

    #taxa_of_interest = get_taxa_of_interest(params['species_set'], params['num_aux_species'], params['aux_species_seed'], params['taxa_file'], taxa_file_snt)

    #locs, labels, _, _, _, _ = load_inat_data(obs_file, taxa_of_interest)
    if params['ts']: 
        locs, labels, _, _, _, timestamps = load_timeseries_data(data_dir)
    else:
        locs, labels, _, _, _ = load_obs_data(data_dir)
    #unique_taxa, class_ids = np.unique(labels, return_inverse=True)
    #class_to_taxa = unique_taxa.tolist()

    # load class names
    #class_info_file = json.load(open(taxa_file, 'r'))
    #class_names_file = [cc['latin_name'] for cc in class_info_file]
    #taxa_ids_file = [cc['taxon_id'] for cc in class_info_file]
    #classes = dict(zip(taxa_ids_file, class_names_file))
    
    idx_ss = get_idx_subsample_observations(labels, params['hard_cap_num_per_class'], params['hard_cap_seed'])

    locs = torch.from_numpy(np.array(locs)[idx_ss]) # convert to Tensor

    #labels = torch.from_numpy(np.array(class_ids)[idx_ss])
    labels = torch.from_numpy(np.array(labels))

    #ds = LocationDataset(locs, labels, classes, class_to_taxa, params['input_enc'], params['device'])
    if params['ts']: 
        ds = TimeLocationDataset(locs, labels, params['input_enc'], params['device'], timestamps)
    else:
        ds = LocationDataset(locs, labels, params['input_enc'], params['device'])
    
    return ds
