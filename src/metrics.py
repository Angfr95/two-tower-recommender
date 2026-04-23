import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_classification_metrics(all_labels, all_scores, threshold=0.5):
    """
    Métriques de classification binaire — évalue la qualité du score brut du modèle.
    On seuille les scores à 0.5 pour obtenir des prédictions 0/1.
    """
    preds = (all_scores >= threshold).astype(float)

    # Accuracy : part des prédictions correctes (toutes classes confondues)
    accuracy = (preds == all_labels).mean()

    # Precision : parmi les items prédits positifs, combien le sont vraiment ?
    # Utile pour mesurer le bruit dans les recommandations
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall : parmi tous les vrais positifs, combien a-t-on retrouvés ?
    # Utile pour mesurer si le modèle rate des films que l'utilisateur aurait aimé
    fn = ((preds == 0) & (all_labels == 1)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 : moyenne harmonique de precision et recall
    # Synthèse utile quand les classes sont déséquilibrées
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # AUC-ROC : probabilité que le modèle score un positif plus haut qu'un négatif
    # Ne dépend pas du seuil — mesure la capacité de discrimination globale
    auc_roc = roc_auc_score(all_labels, all_scores)

    # PR-AUC : aire sous la courbe Precision-Recall
    # Plus informative qu'AUC-ROC quand les négatifs sont très majoritaires
    pr_auc = average_precision_score(all_labels, all_scores)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc_roc":   round(auc_roc, 4),
        "pr_auc":    round(pr_auc, 4),
    }


def compute_ranking_metrics(all_labels, all_scores, all_users, k=10):
    """
    Métriques de ranking — évalue la qualité des recommandations top-K par utilisateur.
    Ces métriques sont plus proches de l'usage réel (on recommande une liste ordonnée).
    """
    users = np.unique(all_users)
    precisions, recalls, ndcgs, hits = [], [], [], []

    for user in users:
        mask = all_users == user
        scores_u = all_scores[mask]
        labels_u = all_labels[mask]

        # On ne peut pas calculer de ranking si l'utilisateur n'a aucun positif
        if labels_u.sum() == 0:
            continue

        # Indices des K items les mieux scorés par le modèle
        top_k_idx = np.argsort(scores_u)[::-1][:k]
        top_k_labels = labels_u[top_k_idx]

        n_relevant = labels_u.sum()

        # Precision@K : fraction des K recommandations qui sont réellement pertinentes
        precisions.append(top_k_labels.sum() / k)

        # Recall@K : fraction des items pertinents que l'on retrouve dans le top-K
        recalls.append(top_k_labels.sum() / n_relevant)

        # Hit Rate@K : 1 si au moins un item pertinent est dans le top-K, 0 sinon
        hits.append(1.0 if top_k_labels.sum() > 0 else 0.0)

        # NDCG@K : Normalized Discounted Cumulative Gain
        # Pénalise les positifs trouvés en bas du classement (log(rang+2) au dénominateur)
        # Un item pertinent en position 1 vaut plus qu'en position 10
        dcg = sum(label / np.log2(rank + 2) for rank, label in enumerate(top_k_labels))
        ideal_labels = np.ones(min(int(n_relevant), k))
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(len(ideal_labels)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"precision@{k}": round(np.mean(precisions), 4),
        f"recall@{k}":    round(np.mean(recalls), 4),
        f"hit_rate@{k}":  round(np.mean(hits), 4),
        f"ndcg@{k}":      round(np.mean(ndcgs), 4),
    }


def evaluate(model, loader, device="cpu"):
    """
    Passe le modèle en mode eval sur le loader et collecte labels, scores, user_ids.
    Retourne les deux groupes de métriques (classification + ranking).
    """
    model.eval()
    all_labels, all_scores, all_users = [], [], []

    with torch.no_grad():
        for user_idx, item_idx, label in loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            scores = model(user_idx, item_idx).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(label.numpy())
            all_users.append(user_idx.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_users  = np.concatenate(all_users)

    clf   = compute_classification_metrics(all_labels, all_scores)
    rank  = compute_ranking_metrics(all_labels, all_scores, all_users, k=10)
    return {**clf, **rank}


def print_metrics(metrics: dict, title: str = "Métriques"):
    print(f"\n{'─' * 40}")
    print(f"  {title}")
    print(f"{'─' * 40}")
    for name, value in metrics.items():
        print(f"  {name:<15} {value:.4f}")
    print(f"{'─' * 40}\n")
