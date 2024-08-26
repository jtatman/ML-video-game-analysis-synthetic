from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
from sdv.single_table import CTGANSynthesizer

'''
    We use two methods to generate synthetic data with a Gaussian-Copula synthesizer model, and then use a longer-epoch formalized and modified GAN, the CTGAN.
   A Gaussian Copula synthesizer is faster, more efficient and handles numerical data especially well, while it is limited by its ability to model complex scenarios, generally follows the assumption of normality, and is less effective for categorical data
   A CTGAN handles imbalanced data well, and furthermore interprets complex relationships more thoroughly. Its weaknesses like in the additonal time of proceeding through epochs like a traditional model training and the possibility of mode collapse where the generator in the GAN model fails. 


References:
    Synthetic Data distributions Random, SDV CTGAN, CopulaGAN, GaussianCopula. (n.d.). Windocks.Com. Retrieved August 26, 2024, from https://www.windocks.com/synthetic-data-sdv-distributions
    
'''


data = pd.read_csv("video-game-dataset.csv")

# create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# create standard gaussian copula synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)

# train synthesizer
synthesizer.fit(data)

# generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)

# save data as CSV
synthetic_data.to_csv("video-game-dataset-extended.csv", index=False)

# now, let's use an advanced GAN model to synthesize additional data
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=1000)
synthetic_data.to_csv("video-game-dataset-extended-ctgan.csv", index=False)


