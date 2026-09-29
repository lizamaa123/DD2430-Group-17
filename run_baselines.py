from src.models.baseline_models import PopularityPrior, ItemContentBaseline, ItemItemCooccurrence
from src.evaluation.evaluator import LeaveOneOutEvaluator
import json
import pandas as pd

def load_data():
    # loading the hardcoded synthetic rooms
    with open("data/synthetic_rooms.json", "r") as f:
        rooms_data = json.load(f)
        
    # splitting the rooms into training and testing based on pre-set splits 
    train_rooms = [[str(item) for item in room["item_ids"]] for room in rooms_data if room["split"] == "train"]
    test_rooms = [[str(item) for item in room["item_ids"]] for room in rooms_data if room["split"] == "test"]

    # loading and extracting item id from IKEA data for Item Content Baseline
    df = pd.read_csv("data/processed_ikea_data.csv")
    item_id = pd.to_numeric(df["item_id"], errors='coerce').fillna(0).astype(int).astype(str).tolist()
    
    # Combine the text columns so SentenceBERT has rich descriptions to read
    # e.g., "MALM Beds A high bed frame with 2 storage boxes"
    df["full_text"] = df["name"] + " " + df["category"] + " " + df["short_description"]
    item_description = df["full_text"].tolist()
    
    return train_rooms, test_rooms, item_id, item_description

if __name__ == "__main__":
    print("Loading data...")
    train_rooms, test_rooms, item_id, item_description = load_data()
    
    print(f"Loaded {len(train_rooms)} training rooms and {len(test_rooms)} test rooms.")
    
    K = 10 # nr of guesses
    evaluator = LeaveOneOutEvaluator()

    # ---------------------------------------------------------
    # Evaluate Popularity Prior
    # ---------------------------------------------------------
    print("\nTraining Popularity Prior...")
    pop_model = PopularityPrior()
    pop_model.fit(train_rooms)
    
    print("Evaluating Popularity Prior...")
    pop_recall, pop_ndcg = evaluator.evaluate(pop_model, test_rooms, k=K)
    print(f"Popularity Prior: Recall@{K} = {pop_recall:.4f} | NDCG@{K} = {pop_ndcg:.4f}")

    # ---------------------------------------------------------
    # Evaluate Item-Item Co-occurrence
    # ---------------------------------------------------------
    print("\nTraining Item-Item Co-occurrence...")
    cooc_model = ItemItemCooccurrence()
    cooc_model.fit(train_rooms)
    
    print("Evaluating Item-Item Co-occurrence...")
    cooc_recall, cooc_ndcg = evaluator.evaluate(cooc_model, test_rooms, k=K)
    print(f"Item-Item Co-occ:Recall@{K} = {cooc_recall:.4f} | NDCG@{K} = {cooc_ndcg:.4f}")

    # ---------------------------------------------------------
    # Evaluate Item Content 
    # ---------------------------------------------------------
    print("\nSetting up Item Content Baseline (Loading SentenceBERT)...")
    content_model = ItemContentBaseline(item_id, item_description)
    content_model.fit(train_rooms) # does nothing but keeps API consistent
    
    print("Evaluating Item Content Baseline...")
    content_recall, content_ndcg = evaluator.evaluate(content_model, test_rooms, k=K)
    print(f"Item Content: Recall@{K} = {content_recall:.4f} | NDCG@{K} = {content_ndcg:.4f}")
