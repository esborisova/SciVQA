# <img src="SciVQA_logo.gif" alt="drawing" width="300"/>


# Data

The SciVQA is a corpus of chart images extracted from scientific publications in Computer Science and Computational Linguistics available in arXiv and ACL Anthology. SciVQA is a subset of the two existing datasets:
- __ACL-Fig: A Dataset for Scientific Figure Classification.__

  _Zeba Karishma, Shaurya Rohatgi, Kavya Shrinivas Puranik, Jian Wu, C. Lee Giles._ <img src='https://img.shields.io/badge/arXiv-2023-darkred'> <a href='https://arxiv.org/abs/2301.12293'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/citeseerx/ACL-fig'><img src='https://img.shields.io/badge/Dataset-gold'></a>

- __SciGraphQA: A Large-Scale Synthetic Multi-Turn Question-Answering Dataset for Scientific Graphs.__

  _Shengzhi Li, Nima Tajbakhsh._ <img src='https://img.shields.io/badge/arXiv-2023-darkred'> <a href='https://arxiv.org/abs/2308.03349'><img src='https://img.shields.io/badge/PDF-blue'></a> <a href='https://huggingface.co/datasets/alexshengzhili/SciGraphQA-295K-train?row=0'><img src='https://img.shields.io/badge/Dataset-gold'></a>

The SciVQA comprises 3000 images in .png format and can be dowloaded from Zenodo or HF: links to be added...

## Dataset statistics

| Source data | N of unique papers | N of chart images | N of QA pairs | 
|-------------|--------------------|-------------------|---------------|
|  ACL-Fig    |   530              |   948             |               | 
|  SciGraphQA |   1750             |   2052            |               | 
|  **Total**  |   **2280**         |   **3000**        |               | 


| Chart types        | N in ACL-Fig subset| N in SciGraphQA subset| 
|--------------------|--------------------|-----------------------|
|line chart          |                    |                       |
|bar chart           |                    |                       |
|pie chart           |                    |                       |
|box plot            |                    |                       |
|scatter plot        |                    |                       |
|venn diagram        |                    |                       |
|confusion matrix    |                    |                       |
|pareto              |                    |                       |
|map                 |                    |                       |
|neural networks     |                    |                       |              
|architecture diagram|                    |                       |
|tree                |                    |                       |              
|graph               |                    |                       |
|heat map?           |                    |                       |
|histogram?          |                    |                       |
 


| QA pair types                                       | N in ACL-Fig subset| N in SciGraphQA subset| 
|-----------------------------------------------------|--------------------|-----------------------|
|closed-ended infinite answer set visual              |                    |                       |   
|closed ended infinite answer set non-visual          |                    |                       | 
|closed-ended finite answer set binary visual         |                    |                       | 
|closed-ended finite answer set binary non-visual     |                    |                       | 
|closed-ended finite answer set non-binary visual     |                    |                       | 
|closed-ended finite answer set none-binary non-visual|                    |                       | 
|unanswerable                                         |                    |                       |



Details on the papers distribution per year and venue are availabel under [utils](https://github.com/esborisova/SciVQA/blob/main/src/utils/papers_dist_3000.png).
**Note:** Currently all images and metadata files are stored on Pegasus and HF under private repository.

# Contributors

Ekaterina Borisova (DFKI)

Raia Abu Ahmad (DFKI)
