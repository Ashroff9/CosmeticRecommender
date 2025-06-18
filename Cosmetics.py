# Task 1: Import and inspect the dataset
import pandas as pd
#import numpy as np
#from sklearn.manifold import TSNE
#from bokeh.plotting import figure, show, output_file
#from bokeh.models import ColumnDataSource, HoverTool

# Read CSV
df = pd.read_csv("cosmetics.csv")

# View a sample of 5 rows
print(df.sample(5))

# Count types of product
print(df["Label"].value_counts())

# Task 2: Filter the data
# Filter for "Moisturizer"
moisturizers = df[df["Label"] == "Moisturizer"]
# Now filter for dry skin (assuming the column name is exactly 'dry')
# First, check the column names to be sure
print(df.columns)

# Filter for dry skin
moisturizers_dry = moisturizers[moisturizers["Dry"] == 1].reset_index(drop=True)

# View shape
print("Moisturizers for dry skin:", moisturizers_dry.shape)
# Task 3: Tokenize the ingredients
corpus = []
ingredient_idx = {}
idx = 0

for ing_list in moisturizers_dry['Ingredients']:
    # Convert to lowercase and split by ", "
    tokens = str(ing_list).lower().split(', ')
    corpus.append(tokens)

    for token in tokens:
        if token not in ingredient_idx:
            ingredient_idx[token] = idx
            idx += 1

print(f"Total unique ingredients: {len(ingredient_idx)}")
print(f"Sample ingredients for first product: {corpus[0]}")

# Task 4: Initialize document-term matrix
import numpy as np

M = len(corpus)  # number of products
N = len(ingredient_idx)  # number of unique ingredients

# M x N zero matrix
A = np.zeros((M, N))

# Task 5: Create one-hot encoder function
def oh_encoder(ingredient_list):
    x = np.zeros(N)
    for ingredient in ingredient_list:
        index = ingredient_idx.get(ingredient)
        if index is not None:
            x[index] = 1
    return x

# Task 6: Fill matrix A using one-hot encoding
for i in range(M):
    A[i] = oh_encoder(corpus[i])

print("Document-Term Matrix Shape:", A.shape)

# Task 7 : Reduce dimension of matrix using t-SNE
from sklearn.manifold import TSNE

# Create TSNE model instance with given parameters
model = TSNE(n_components=2, learning_rate=200, random_state=42)

# Fit and transform the document-term matrix A
tsne_features = model.fit_transform(A)

# Add the two t-SNE components as new columns in moisturizers_dry DataFrame
moisturizers_dry['X'] = tsne_features[:, 0]
moisturizers_dry['Y'] = tsne_features[:, 1]

# Check first few rows with new columns
print(moisturizers_dry[['Name', 'X', 'Y']].head())

#Task 8 : Plot scattered plot with vectorized items
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# Prepare data source for Bokeh
source = ColumnDataSource(moisturizers_dry)

# Create figure
plot = figure(title="t-SNE visualization of Moisturizers for Dry Skin",
              x_axis_label='T-SNE 1', y_axis_label='T-SNE 2',
              tools="pan,wheel_zoom,box_zoom,reset")


plot.scatter(x='X', y='Y', size=8, source=source, fill_alpha=0.6)

#Task 9: Add hover tool
from bokeh.models import HoverTool

# Define tooltips content
tooltips = [
    ('Item', '@Name'),
    ('Brand', '@Brand'),
    ('Price', '$@Price'),
    ('Rank', '@Rank'),
]

hover = HoverTool(tooltips=tooltips)
plot.add_tools(hover)

#Task 10:  Show plot
show(plot)

#Task 11: Compare two products
# Filter products by their exact names
product_1 = moisturizers_dry[moisturizers_dry['Name'] == 'Color Control Cushion Compact Broad Spectrum SPF 50+']
product_2 = moisturizers_dry[moisturizers_dry['Name'] == 'BB Cushion Hydra Radiance SPF 50']

# Print details
print("Product 1 Details:")
print(product_1[['Name', 'Brand', 'Price', 'Rank', 'Ingredients']].to_string(index=False))
print("\nProduct 1 Ingredients:")
print(product_1['Ingredients'].values[0])

print("\n---------------------------------------\n")

print("Product 2 Details:")
print(product_2[['Name', 'Brand', 'Price', 'Rank', 'Ingredients']].to_string(index=False))
print("\nProduct 2 Ingredients:")
print(product_2['Ingredients'].values[0])
output_file("Bokeh_plot.html")
show(plot)

