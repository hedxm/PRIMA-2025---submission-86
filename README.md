
This archive contains the complete source code and required files. This guide provides step-by-step instructions to set up and run the knowledge graph construction pipeline.


## Contents of this Archive

*   `/agents`: Core Python logic for each LLM agent.
*   `/configs`: All JSON configuration files for the system.
*   `/data`: Contains a placeholder for the Numberbatch embeddings file.
*   `/files`: Contains the source PDF documents used for the case study.
*   `/prompts`: Contains the prompt templates used by the LLM agents.
*   `/utils` & `/tools`: Various utility and helper functions.
*   `initial_schema.backup`: The Neo4j database backup with the foundational ontology.
*   `/outputs`: Contains the outputs of each component of the pipeline, inside of the LLM agents outputs there is also a folder containing the logs.
*   `main.py`: The main script to execute the pipeline.
*   `requirements.txt`: A list of the required Python dependencies.
*   `README.md`: This instruction file.


## Setup Instructions

Please follow these steps in order to run the project.

### Prerequisites

*   Python 3.9+
*   Neo4j Desktop
*   Ollama
*   **Numberbatch Embeddings**: This project requires a large data file containing commonsense knowledge vectors. This file is used by the `Commonsense Verifier` module to check if an extracted fact is plausible.

**Note:** Due to its very large size (over 3 GB), this file is not included in the repository and must be downloaded manually.

**Instructions to get the file:**

1.  **Download:** Go to the official [ConceptNet Downloads page](https://conceptnet.io/downloads).
2.  **Select the Correct File:** Find and download the English embeddings. The exact filename you need is **`numberbatch-en-19.08.txt`**.
3.  **Place the File:** After downloading, move the `numberbatch-en-19.08.txt` file into the `/data` directory within the main project folder.

Once this is done, you can proceed with the setup steps below.

### Step 1: Set up the Neo4j Database

This step is critical as it loads the required foundational ontology.

1.  Open **Neo4j Desktop**.
2.  Create a new, empty Project.
3.  Inside the project, create a new Database Management System (DBMS). Use any password you like.
4.  Once the DBMS is created, click the "..." menu next to it and select **Administration**.
5.  Navigate to the **Backup** tab.
6.  Click the **Restore** button.
7.  In the restore dialog, select the **From dump** option and browse to the `initial_schema.backup` file from the cloned repository.
8.  Complete the restore process.
9.  **Start** the database. Your Neo4j instance is now ready.

Note: This file can also be used on Neo4j Aura Database (cloud based).

1.  Go to the [Neo4j Aura]([https://neo4j.com/cloud/aura-graph-database/](https://console.neo4j.io/)) website and create a new **Free** instance.
2.  Once your instance is active, open the **Neo4j Browser** for that instance.
3. Once the instance is created, click the "..." menu next to it and select **Backup & restore**.
4. In the restore dialog, select the **From dump** option and browse to the `initial_schema.backup` file included in this archive.
8.  Complete the restore process.
9.  **Start** the database. Your Neo4j instance is now ready.

### Step 2: Clone the Repository and Set up the Environment

1.  Open a terminal or command prompt.
2.  Clone the repository from GitHub:
    ```sh
    git clone https://github.com/hedxm/PRIMA-2025---submission-86.git
    ```
3.  Navigate into the newly created project folder:
    ```sh
    cd PRIMA-2025---submission-86
    ```
4.  Create and activate a Python virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
5.  Install all required Python dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Step 3: Set up the Local LLM

1.  Ensure the Ollama application is running in the background on your machine.
2.  Pull the required LLM model by running this command in your terminal:
    ```sh
    ollama pull llama3:instruct
    ```

## Configuration

The final step is to connect the Python script to your Neo4j database.

1.  Open the following two files in a text editor:
    *   `configs/triplet_matcher.json`
2.  In the file, find the `"neo4j"` configuration block.
3.  Update the `uri`, `user`, and `password` fields to match the credentials of the Neo4j database you set up in Step 1.

    ```json
    // Example configuration block
    "neo4j": {
      "uri": "bolt://localhost:7687",
      "user": "neo4j",
      "password": "your_password_here"
    }
    ```
4.  The pipeline is already configured to use the documents in the `/files` directory, so no other changes are necessary.

## Customizing the KG Curator for a New Domain (Advanced)

The MAKG pipeline can be adapted to other technical domains. The most important component to customize is the `KG Curator`, which is responsible for assigning the correct labels to entities. This is controlled by two key files.

### Step 1: Modifying the Target Labels in `kg_curator.json`

The official list of valid Neo4j node labels is defined in `configs/kg_curator.json`. To adapt the system, you must edit the `target_kg_labels` list in this file. You can add, remove, or change labels to fit your target domain. The `default_label` ("Review") is used as a fallback for any entity the agent cannot confidently classify.

```json
// From configs/kg_curator.json
"target_kg_labels": [
    "System", "Sensor", "Actuator", "Controller",
    // ... (add, remove, or change labels here)
    "Feature", "Hardware"
],
"default_label": "Review",
```

### Step 2: Updating the Prompt Examples in `kg_curator.txt`

After changing the list of labels, you **must** update the examples in the prompt to teach the LLM how to use them. This is crucial for the agent's accuracy.

1.  Open `prompts/kg_curator.txt`.
2.  Find the `--- EXAMPLES ---` section.
3.  Modify the `INPUT ENTITY NAMES` and `OUTPUT JSON DICTIONARY` pairs to reflect your new labels. These examples show the agent how to handle direct mappings, generalizations, and ambiguous terms for your specific domain.

For example, if you add a new label called `Requirement`, you should add a relevant term to an `INPUT` list and show its correct classification in the `OUTPUT` dictionary, like this:

INPUT ENTITY NAMES:
["LiDAR", "Automated Driving System", "FR-101", "Thingamajig"]

OUTPUT JSON DICTIONARY:
{"LiDAR": "LiDAR", "Automated Driving System": "System", "FR-101": "Requirement", "Thingamajig": "Review"}


## Running the Pipeline

**IMPORTANT:** Before starting a new run, it is highly recommended to clear the contents of the folders inside of the `/outputs` directory, do not delete any folder whatsoever (`/logs`), just files. This ensures that results from previous executions do not overlap with new ones and provides a clean state for your experiment.

Once the setup and configuration are complete, execute the pipeline by running the `main.py` script from the project's root directory:

```sh
python main.py
```

The terminal will display detailed log messages as the system progresses through each stage. The process may take some time to complete depending on your computer's hardware.
