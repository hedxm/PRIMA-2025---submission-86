import json

def load_triplets_from_json(filepath):
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    triplets = [] 
    
    for item in data:
        head = item['subject'].strip()
        relation = item['predicate'].strip()
        tail = item['object'].strip()
        triplets.append((head,relation,tail))
        
        return triplets
    
    
    