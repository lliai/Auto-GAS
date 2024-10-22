# Import the module to be tested
from gaswot.tnb101.model_builder import create_model

m = create_model('64-41414-3_33_212', 'autoencoder')

print(m)
