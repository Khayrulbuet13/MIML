# Multiplex Image Machine Learning (MIML)

The Multiplex Image Machine Learning (MIML) framework stands at the forefront of advancing cell classification methodologies. This state-of-the-art model synergizes label-free cell imaging with biomechanical property data, capturing the rich morphological nuances inherent in each cell. By embracing a dual-data approach, MIML transcends the capabilities of conventional machine learning architectures, ensuring that no morphological information is overlooked.

With an unprecedented 98.3% accuracy in cell classification, MIML heralds a new era in precision diagnostics and cellular analysis. Its utility has been validated in the classification of white blood cells and tumor cells, showcasing a profound potential for widespread application. The inherent adaptability and transfer learning prowess of MIML make it a prime candidate for extending its benefits to other cell types, especially those with similar morphology yet distinct biomechanical characteristics.

MIML's innovative integration promises transformative impacts across diverse domains, catalyzing breakthroughs in disease diagnosis and deepening our comprehension of cellular dynamics.


![Alt text](Images/MIML.svg)


## File Structure
For seamless project integration, MIML is systematically organized as below:


```txt
MIML/
├── main.py                  # Code for training the model
├── utils.py                 # Utility functions for data manipulation and visualization
├── models/
│   ├── __init__.py
│   └── MIML.py              # Model architectures and training routines
├── data/
│   ├── __init__.py
│   ├── MyDataset.py         # Custom dataset class for MIML
│   └── transformation/
│       └── __init__.py      # Data augmentation routines
├── notebooks/
│   └── example_usage.ipynb  # Jupyter notebooks illustrating MIML's dataset and model usage
├── callbacks/
│   ├── __init__.py
│   └── CometCallback.py     # Callbacks for comet_ml integration
├── .gitignore
├── logger.py
├── project.py               # File structure definition
├── README.md
├── secrets.json             # secret credentials for comet_ml
└── LICENSE

```

<!-- Getting Started with MIML -->
## Getting Started with MIML

Embark on your journey with the MIML framework through this simplified setup guide. This will guide you through the installation of dependencies, setting up the MIML package, and preparing the data for your machine learning endeavors.

### Install Dependencies
```bash
pip install -r requirements.txt
```


### Built With

This project is built with the following frameworks and libraries, which are integral to its development and functioning:

* [![PyTorch][PyTorch-shield]][PyTorch-url]
* [![imgaug][imgaug-shield]][imgaug-url]
* [![NumPy][numpy-shield]][numpy-url]
* [![Torchvision][Torchvision-shield]][Torchvision-url]
* [![Comet ML][CometML-shield]][CometML-url]
* [![Matplotlib][matplotlib-shield]][matplotlib-url]
* [![Poutyne][Poutyne-shield]][Poutyne-url]
* [![Pandas][pandas-shield]][pandas-url]





<!-- LICENSE -->
## License

Distributed under the Apache License. See `LICENSE` for more information.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](./LICENSE)





<!-- CONTACT -->
## Contact

Khayrul Islam - [@LinkedIN](https://linkedin.com/in/khayrulbuet13) - khayrulbuet13@alum.lehigh.edu

Project Link: [https://github.com/Khayrulbuet13/MIML](https://github.com/Khayrulbuet13/MIML)



<!-- ACKNOWLEDGMENTS -->
<br><br>
## Acknowledgments

This project is funded by:

![NIH](Images/NSF.jpeg)




<!-- Release Note -->

## Release Note

*0.1.0 - Initial Release: Unveiling of MIML, a game-changer in label-free cell classification with integration of biomechanical properties.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->

<!-- Linkedin -->
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-blue?logo=linkedin

[linkedin-url]: https://linkedin.com/in/khayrulbuet13


<!-- Pytorch -->
[PyTorch-shield]:https://img.shields.io/static/v1?style=for-the-badge&message=PyTorch&color=EE4C2C&logo=PyTorch&logoColor=FFFFFF&label=

[PyTorch-url]:https://pytorch.org


<!-- NumPy -->
[NumPy-shield]: https://img.shields.io/static/v1?style=for-the-badge&message=NumPy&color=013243&logo=NumPy&logoColor=FFFFFF&label=

[NumPy-url]: https://numpy.org

<!-- tqdm -->
[tqdm-shield]:  https://img.shields.io/static/v1?style=for-the-badge&message=tqdm&color=222222&logo=tqdm&logoColor=FFC107&label=

[tqdm-url]: https://tqdm.github.io


<!-- Matplotlib -->
[Matplotlib-shield]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org



<!-- imgaug -->
[imgaug-shield]: https://img.shields.io/badge/imgaug-776AB?style=for-the-badge&logo=imgaug&logoColor=white
[imgaug-url]: https://imgaug.readthedocs.io/en/latest/

<!-- Torchvision -->
[Torchvision-shield]: https://img.shields.io/badge/Torchvision-06bcee?style=for-the-badge&logo=PyTorch&logoColor=white
[Torchvision-url]: https://pytorch.org/vision/stable/index.html

<!-- Comet ML -->
[CometML-shield]: https://img.shields.io/badge/CometML-9cf?style=for-the-badge&logo=CometML&logoColor=white
[CometML-url]: https://www.comet.ml

<!-- Poutyne -->
[Poutyne-shield]: https://img.shields.io/badge/Poutyne-4caf50?style=for-the-badge&logo=Poutyne&logoColor=white
[Poutyne-url]: https://poutyne.org

<!-- Pandas -->
[Pandas-shield]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org