from neo4j import GraphDatabase
import csv
import os

def export_neo4j_triplets(uri, user, password, output_path):
    """Exports all S-P-O triplets from Neo4j to a CSV file."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with driver.session() as session:
        # Assuming nodes have a 'name' property for subject and object.
        # Adjust n.name, m.name if your property key is different.
        result = session.run("""
            MATCH (n)-[r]->(m)
            WHERE n.name IS NOT NULL AND m.name IS NOT NULL
            RETURN n.name AS head, type(r) AS relation, m.name AS tail
        """)
        
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t') # Using tab delimiter as in your precompute_embeddings
            writer.writerow(["subject", "predicate", "object"]) # Header
            for row in result:
                writer.writerow([row['head'], row['relation'], row['tail']])
    
    print(f"✅ Exported triplets to: {output_path}")
    driver.close()

def export_neo4j_entities(uri, user, password, output_path):
    """Exports unique entity names from Neo4j to a CSV file."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_entities = set()
    
    with driver.session() as session:
        # Adjust n.name if your entity identifier property is different
        result = session.run("MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS entity_name")
        for record in result:
            if record["entity_name"]: # Ensure it's not None or empty
                unique_entities.add(record["entity_name"])
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f) # Default CSV delimiter (comma) is fine here
        writer.writerow(["entity_name"]) # Header
        for entity in sorted(list(unique_entities)): # Sort for consistent output
            writer.writerow([entity])
            
    print(f"✅ Exported {len(unique_entities)} unique entities to: {output_path}")
    driver.close()

def export_neo4j_relationship_types(uri, user, password, output_path):
    """Exports unique relationship types from Neo4j to a CSV file."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_relationship_types = set()
    
    with driver.session() as session:
        result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        for record in result:
            if record["relationshipType"]: # Ensure it's not None or empty
                unique_relationship_types.add(record["relationshipType"])

    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f) # Default CSV delimiter (comma) is fine here
        writer.writerow(["relationship_type"]) # Header
        for rel_type in sorted(list(unique_relationship_types)): # Sort
            writer.writerow([rel_type])
            
    print(f"✅ Exported {len(unique_relationship_types)} unique relationship types to: {output_path}")
    driver.close()

