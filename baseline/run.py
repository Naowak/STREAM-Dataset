# run_experiments.py
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import stream_dataset as sd
from models import DynamicLSTM

def get_masked_loss(preds, targets, timesteps, category):
    """
    Calcule la loss uniquement sur les timesteps indiqués par la tâche.
    """
    B = timesteps.shape[0]
    O = preds.shape[-1]
    
    # Extraire les prédictions et cibles aux timesteps spécifiques
    # list comprehension pour parcourir le batch et indexer correctement
    p_list = [preds[i, timesteps[i], :] for i in range(B)]
    t_list = [targets[i, timesteps[i], :] for i in range(B)]
        
    p_tensor = torch.stack(p_list) # [B, P, O]
    t_tensor = torch.stack(t_list) # [B, P, O]
    
    if category == 'classification':
        # CrossEntropyLoss attend (N, C) et target (N) comme indices de classes
        t_class = torch.argmax(t_tensor, dim=-1) 
        p_flat = p_tensor.view(-1, O)            
        t_flat = t_class.view(-1)                
        loss = nn.CrossEntropyLoss()(p_flat, t_flat)
        
    elif category == 'multi_classification':
        # Multi-label attend des probas/logits sur toutes les dimensions
        p_flat = p_tensor.view(-1, O)
        t_flat = t_tensor.view(-1, O)
        loss = nn.BCEWithLogitsLoss()(p_flat, t_flat)
        
    elif category == 'regression':
        loss = nn.MSELoss()(p_tensor, t_tensor)
        
    return loss

def run_experiment():
    parser = argparse.ArgumentParser(description="Run Stream Dataset Evaluation on PyTorch")
    parser.add_argument('--tasks', nargs='+', default=['simple_copy', 'adding_problem'], help='Liste des tâches (ex: sinus_forecasting bracket_matching)')
    parser.add_argument('--difficulties', nargs='+', default=['small'], choices=['small', 'medium', 'large'], help='Niveaux de difficulté')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 10000], help='Tailles des paramètres visées (ex: 1000 10000 100000)')
    parser.add_argument('--seeds', type=int, default=5, help='Nombre de seeds par run')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Taille du batch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32', help='Type de données Pytorch (float32, float64)')
    parser.add_argument('--output', type=str, default='results.csv', help='Fichier CSV de sortie')
    
    args = parser.parse_args()
    
    # Mapping du dtype
    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    
    results = []
    
    combinations = list(itertools.product(args.tasks, args.difficulties, args.sizes, range(args.seeds)))
    
    for task_name, difficulty, size, seed in combinations:
        print(f"\n--- Running Task: {task_name} | Diff: {difficulty} | Size: {size} | Seed: {seed} ---")
        
        # 1. Charger les données du Stream Dataset
        try:
            task_data = sd.build_task(task_name, difficulty=difficulty, seed=seed)
        except Exception as e:
            print(f"Erreur lors du chargement de {task_name} ({difficulty}): {e}")
            continue
        
        category = task_data['category']
        
        # Conversion en Tensor PyTorch
        X_train = torch.tensor(task_data['X_train'], dtype=torch_dtype)
        Y_train = torch.tensor(task_data['Y_train'], dtype=torch_dtype)
        T_train = torch.tensor(task_data['T_train'], dtype=torch.long) # indices entiers
        
        X_test = torch.tensor(task_data['X_test'], dtype=torch_dtype).to(device)
        
        # Dataloader pour le training
        train_dataset = TensorDataset(X_train, Y_train, T_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Dimensions
        input_dim = X_train.shape[-1]
        output_dim = Y_train.shape[-1]
        
        # 2. Instanciation dynamique
        model = DynamicLSTM(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print(f"Modèle LSTM initialisé avec ~{model.actual_params} paramètres (Cible: {size}).")
        
        # 3. Boucle d'entrainement
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            for bx, by, bt in train_loader:
                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                
                optimizer.zero_grad()
                preds = model(bx)
                
                # Appliquer le masque de perte spécifique
                loss = get_masked_loss(preds, by, bt, category)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # 4. Évaluation sur le test set
        model.eval()
        with torch.no_grad():
            preds_test = model(X_test)
            # Retourner les predictions en numpy pour sd.compute_score
            preds_test_np = preds_test.cpu().numpy()
            
        # Appel de l'évaluation native du package
        score = sd.compute_score(
            Y=task_data['Y_test'],
            Y_hat=preds_test_np,
            prediction_timesteps=task_data['T_test'],
            category=category
        )
        print(f"Score de Test (Lower is better): {score:.4f}")
        
        # 5. Stockage des résultats
        results.append({
            'Task': task_name,
            'Difficulty': difficulty,
            'Target_Params': size,
            'Actual_Params': model.actual_params,
            'Seed': seed,
            'Category': category,
            'Test_Score': score
        })
        
    # Exporter en CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nExpériences terminées ! Résultats sauvegardés dans {args.output}")

if __name__ == "__main__":
    run_experiment()