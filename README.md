# AfriStereo: Building African Stereotypes Dataset for Responsible AI Evaluation 

This repository contains the code-base used for the Afri-Stereo project. This includes the complete pipeline (manual + automated) to generate the stereotype dataset, and also the code used to perform the various LLM evaluations. 

## Getting Started

1. Clone the Repository

```bash
git clone https://github.com/yourname/yourrepo.git
```

2. Install all requirements.

Note: Preferably create an anaconda environment before doing so. 

```bash
pip install -r requirements.txt
```

## Understanding the Structure of the Repository

```text
afristereo/
├── data/                   
│   ├── raw/                # Raw/Input Data
│   └── processed/          # Processed/Output Data
├── app/                    # Streamlit annotation interface
├── evaluation/             # Scripts for LLM evaluation
├── scripts/                # Scripts for Processing Data and Outputs
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── LICENSE                 # Project license
```

