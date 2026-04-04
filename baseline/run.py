# run_experiments.py
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy

import sys
sys.path.append('../')  # Permet d'importer stream_dataset et models depuis le dossier parent
import stream_dataset as sd
from models import DynamicLSTM, DynamicGRU, DynamicTransformerDecoderOnly, DynamicTransformerEncoderDecoder

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_masked_loss(preds, targets, timesteps, category):
    """
    Calcule la loss uniquement sur les timesteps indiqués par la tâche.
    """
    B = timesteps.shape[0]
    O = preds.shape[-1]
    
    p_list = [preds[i, timesteps[i], :] for i in range(B)]
    t_list = [targets[i, timesteps[i], :] for i in range(B)]
        
    p_tensor = torch.stack(p_list)
    t_tensor = torch.stack(t_list)
    
    if category == 'classification':
        t_class = torch.argmax(t_tensor, dim=-1) 
        p_flat = p_tensor.view(-1, O)            
        t_flat = t_class.view(-1)                
        loss = nn.CrossEntropyLoss()(p_flat, t_flat)
        
    elif category == 'multi_classification':
        p_flat = p_tensor.view(-1, O)
        t_flat = t_tensor.view(-1, O)
        loss = nn.BCEWithLogitsLoss()(p_flat, t_flat)
        
    elif category == 'regression':
        loss = nn.MSELoss()(p_tensor, t_tensor)
        
    return loss

def find_best_threshold(preds_logits, targets, timesteps):
    """
    Trouve le seuil optimal sur le set de validation pour minimiser l'Error Rate.
    """
    B = timesteps.shape[0]
    p_list = [preds_logits[i, timesteps[i], :] for i in range(B)]
    t_list = [targets[i, timesteps[i], :] for i in range(B)]
    
    preds = np.stack(p_list, axis=0)
    truths = np.stack(t_list, axis=0)
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    preds_probs = sigmoid(preds)
    
    best_thresh = 0.5
    best_score = float('inf')
    
    # On teste 100 seuils entre 0.01 et 0.99
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for t in thresholds:
        preds_bin = (preds_probs >= t).astype(int)
        correct_samples = np.all(preds_bin == truths, axis=(1, 2))
        score = 1 - np.mean(correct_samples)
        
        if score < best_score:
            best_score = score
            best_thresh = t
            
    return best_thresh

def run_experiment():
    parser = argparse.ArgumentParser(description="Run Stream Dataset Evaluation on PyTorch")
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'gru', 'transformer_decoder', 'transformer_encdec'], help='Type de modèle à entrainer')
    parser.add_argument('--tasks', nargs='+', default=['all'], help='Liste des tâches')
    parser.add_argument('--difficulties', nargs='+', default=['small'], choices=['small', 'medium', 'large'], help='Niveaux de difficulté')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 10000], help='Tailles des paramètres visées')
    parser.add_argument('--seeds', type=int, default=5, help='Nombre de seeds par run')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Taille du batch')
    parser.add_argument('--patience', type=int, default=10, help='Nombre d\'epochs sans amélioration avant d\'arrêter (0 pour désactiver)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32', help='Type de données Pytorch')
    parser.add_argument('--output', type=str, default='results.csv', help='Fichier CSV de sortie')
    
    args = parser.parse_args()

    if args.tasks == ['all']:
        args.tasks = sd.tasks
    
    torch_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    
    results = []
    
    combinations = list(itertools.product(args.tasks, args.difficulties, args.sizes, range(args.seeds)))
    
    for task_name, difficulty, size, seed in combinations:
        print(f"\n--- Model: {args.model_type.upper()} | Task: {task_name} | Diff: {difficulty} | Size: {size} | Seed: {seed} ---")
        
        try:
            task_data = sd.build_task(task_name, difficulty=difficulty, seed=seed)
        except Exception as e:
            print(f"Erreur lors du chargement de {task_name} ({difficulty}): {e}")
            continue

        set_seed(seed)
        category = task_data['category']
        
        # Train data
        X_train = torch.tensor(task_data['X_train'], dtype=torch_dtype)
        Y_train = torch.tensor(task_data['Y_train'], dtype=torch_dtype)
        T_train = torch.tensor(task_data['T_train'], dtype=torch.long)
        
        # Validation data (On les garde en numpy pour sd.compute_score, mais X_valid passe sur le GPU)
        X_valid = torch.tensor(task_data['X_valid'], dtype=torch_dtype).to(device)
        Y_valid_np = task_data['Y_valid']
        T_valid_np = task_data['T_valid']
        
        # Test data
        X_test = torch.tensor(task_data['X_test'], dtype=torch_dtype).to(device)
        
        train_dataset = TensorDataset(X_train, Y_train, T_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        input_dim = X_train.shape[-1]
        output_dim = Y_train.shape[-1]
        
        # Sélection du modèle
        if args.model_type == 'lstm':
            model = DynamicLSTM(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
        elif args.model_type == 'gru':
            model = DynamicGRU(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
        elif args.model_type == 'transformer_decoder':
            model = DynamicTransformerDecoderOnly(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
        elif args.model_type == 'transformer_encdec':
            model = DynamicTransformerEncoderDecoder(input_dim=input_dim, output_dim=output_dim, target_params=size).to(device, dtype=torch_dtype)
        else:
            raise ValueError(f"Unknown model type {args.model_type}.")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print(f"Modèle {args.model_type.upper()} initialisé avec ~{model.actual_params} paramètres (Cible: {size}).")
        
        # --- Variables pour l'Early Stopping ---
        best_val_score = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            # Phase d'entrainement
            model.train()
            total_loss = 0.0
            for bx, by, bt in train_loader:
                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                
                optimizer.zero_grad()
                preds = model(bx)
                
                loss = get_masked_loss(preds, by, bt, category)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                
            # Phase de validation (à la fin de chaque epoch)
            model.eval()
            with torch.no_grad():
                preds_val = model(X_valid)
                preds_val_np = preds_val.cpu().numpy()
                
            val_score = sd.compute_score(
                Y=Y_valid_np,
                Y_hat=preds_val_np,
                prediction_timesteps=T_valid_np,
                category=category
            )
            
            # Plus le score est bas, meilleur c'est (Error rate ou MSE)
            if val_score < best_val_score:
                best_val_score = val_score
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                
            # Check de l'early stopping
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"  -> Early stopping déclenché à l'epoch {epoch+1}/{args.epochs} (Meilleur score val: {best_val_score:.4f})")
                break
                
        # Restauration des meilleurs poids si on a utilisé l'early stopping
        if args.patience > 0 and best_model_state is not None:
            model.load_state_dict(best_model_state)
                
        # Si c'est du multi-label, on optimise le seuil sur la validation !
        best_threshold = 0.5
        if category == 'multi_classification':
            model.eval()
            with torch.no_grad():
                preds_val = model(X_valid).cpu().numpy()
            best_threshold = find_best_threshold(preds_val, Y_valid_np, T_valid_np)
            print(f"  -> Meilleur threshold trouvé sur la validation : {best_threshold:.2f}")

        # Phase de test final
        model.eval()
        with torch.no_grad():
            preds_test = model(X_test)
            preds_test_np = preds_test.cpu().numpy()
            
        score = sd.compute_score(
            Y=task_data['Y_test'],
            Y_hat=preds_test_np,
            prediction_timesteps=task_data['T_test'],
            category=category,
            threshold=best_threshold
        )
        print(f"Score de Test final: {score:.4f}")
        
        results.append({
            'Model': args.model_type.upper(),
            'Task': task_name,
            'Difficulty': difficulty,
            'Target_Params': size,
            'Actual_Params': model.actual_params,
            'Seed': seed,
            'Category': category,
            'Best_Val_Score': best_val_score,
            'Test_Score': score
        })
        
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nExpériences terminées ! Résultats sauvegardés dans {args.output}")

if __name__ == "__main__":
    run_experiment()