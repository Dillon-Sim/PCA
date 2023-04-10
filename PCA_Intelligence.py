import pandas as pd
import numpy as np
import os 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load in data
df_intel = pd.read_excel("Intelligence.xls")
#extract feature names
feature_names = df_intel.columns
intel_mat = np.array(df_intel)
#examine correlation 
intel_cov = np.corrcoef(intel_mat.T)


# create a StandardScaler object
scaler = StandardScaler()

# fit and transform the data
intel_scaled = scaler.fit_transform(intel_mat)

# Initialize lists to store the output
explained_variance_ratio = []
cumulative_variance_ratio = []
eigen_values = []

#We want to identify the optimal number of PCs to use
#Calculate the max number of PCs
max_pcs = len(feature_names)
max_pca = PCA(n_components= max_pcs)
max_pcs_result = max_pca.fit_transform(intel_scaled)
max_pc_names = ['PC'+str(i) for i in range(1,max_pcs+1)]
PC_values = np.arange(max_pca.n_components_) + 1
max_pca_df = pd.DataFrame(data = max_pcs_result, columns = max_pc_names)

#Scree plot
plt.plot(PC_values, max_pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

print ("Proportion of Variance Explained : ", max_pca.explained_variance_ratio_)  
print ("Cumulative Prop. Variance Explained: ", np.cumsum(max_pca.explained_variance_ratio_)  )

#We can observe that we should choose 3 PCs to explain approximately 60% of the variance

# Perform PCA on the data with 3 principal components
pca = PCA(n_components = 3)
pca.fit(intel_mat)

# Extract the loadings, proportion of variance explained, and singular values
loadings = pca.components_
variance_explained = pca.explained_variance_ratio_
singular_values = pca.singular_values_

# Create new column names for the PCs
pc_names = ['PC'+str(i) for i in range(1,4)]

# Create a table for the loadings
loadings_table = pd.DataFrame(loadings.T, columns=pc_names, index=df_intel.columns)
# Replace values below 0.3 with empty strings
loadings_table = loadings_table.applymap(lambda x: '' if abs(x) < 0.3 else x)# Print the loadings table
print("Loadings:\n", loadings_table)

# Get principal component scores for data
pc_scores = pca.transform(intel_mat)

# Convert the PCs to a DataFrame and assign the new column names
df_pcs = pd.DataFrame(pc_scores, columns=pc_names)

# Concatenate the original DataFrame and the PC DataFrame along columns axis
df_combined = pd.concat([df_intel, df_pcs], axis=1)
