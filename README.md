
KTH project in DD2430 (Group 17): Investigating room-conditioned recommendation systems using set encoders and PyTorch to predict missing room items for IKEA.


Setup Instructions:
1. Clone the repository: `git clone [URL]`
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

Using the complete item catalog
--------------------------------

`src/data_generation/item_encoder.py` builds the same text used when creating
the Sentence-BERT embeddings. The catalog loader keeps item IDs, descriptions,
and embeddings in one consistent order:

```python
from src.data_generation.item_encoder import (
    load_aligned_embeddings,
    load_item_catalog,
)
from src.models.transformer import FurnitureEmbeddingPredictor, TransformerEvaluator

item_ids, item_descriptions = load_item_catalog()
item_embeddings = load_aligned_embeddings(item_ids)
id_to_description = dict(zip(item_ids, item_descriptions))

model = FurnitureEmbeddingPredictor()
evaluator = TransformerEvaluator(
    model,
    item_descriptions=item_descriptions,
    item_embeddings=item_embeddings,
)

# Convert generated room item IDs to the same text representation.
room_item_ids = [item_ids[0], item_ids[1]]
room = [id_to_description[item_id] for item_id in room_item_ids]
predictions = evaluator.predict(room, k=10)
```

If `data/item_embeddings.pt` has not been generated yet, omit
`item_embeddings` and the evaluator will encode the catalog once itself. To
generate the file, run `python -m src.data_generation.item_encoder`.


Collaboration & Workflow:

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for our rules on branch naming conventions, linking commits to issues and our team merge process.