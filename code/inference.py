import argparse
import torch
from models import ResidualFCNet, ResidualFCNetLatent  
from inference_utils import inference_basic, inference_with_latent  # 위에서 정의한 함수들
import os
import datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--use-latent', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-path', type=str, default='inference_results.pt')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    params = {
        'ts': True,                              # 시계열 예측 여부
        'input_enc': 'sin_cos_env',              # 입력 인코딩 방식 ('raw', 'env', 'sin_cos', 'sin_cos_env' 등)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 64,                        # inference 시 배치 크기
        'input_dim': 6,                          # temporal encoding 차원 수 + 위치 인코딩 등 (예: y,m,d,h,min,s)
        'log_frequency': 100,                    # log 출력 빈도
    }

    # Load data
    inference_dataset = datasets.get_inference_data(params)
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['model_args']
    
    if args.use_latent: 
        model = ResidualFCNetLatent(**model_args).to(device)
    else:
        model = ResidualFCNet(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference
    if args.use_latent:
        print("Running latent-based inference...")
        preds = inference_with_latent(model, inference_loader, device)
    else:
        print("Running basic inference...")
        preds = inference_basic(model, inference_loader, device)

    # Save predictions
    torch.save(preds, args.output_path)
    print(f"Saved predictions to {args.output_path}")


if __name__ == '__main__':
    main()