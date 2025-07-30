
def flat_triplets_util(triplets):
    expanded = []
    for triplet in triplets:
        subject = triplet["subject"]
        predicate = triplet["predicate"]
        obj = triplet["object"]

        if isinstance(obj, list):
            for o in obj:
                expanded.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": o
                })
        else:
            expanded.append(triplet)

    return expanded
