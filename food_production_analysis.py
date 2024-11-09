import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objs as go
import altair as alt
from streamlit_option_menu import option_menu








# Read data
df = pd.read_csv("FAO.csv", encoding="ISO-8859-1")
df_pop = pd.read_csv("FAOSTAT_data_6-13-2019.csv")
df_area = pd.read_csv("countries_area_2013.csv")

# Mempersiapkan data untuk area tertentu
def prepare_data_for_area(area_name):
    d3 = df.loc[:, 'Y1993':'Y2013']
    data1 = new_data.join(d3)

    d4 = data1.loc[data1['Element'] == 'Food'] 
    d5 = d4.drop('Element', axis=1)
    d5 = d5.fillna(0)

    # Membuat daftar tahun
    year_list = list(d3.columns)

    # Mendapatkan data untuk area yang dipilih
    selected_area = d4[d4['Area'] == area_name]
    selected_area_total = selected_area.groupby('Item')[year_list].sum()
    selected_area_total['Total'] = selected_area_total.sum(axis=1)
    selected_area_total = selected_area_total.reset_index()

    return selected_area_total

# Penyesuaian konsistensi data
df['Area'] = df['Area'].replace(['Swaziland'], 'Eswatini')
df['Area'] = df['Area'].replace(['The former Yugoslav Republic of Macedonia'], 'North Macedonia')

df_pop = pd.DataFrame({'Area': df_pop['Area'], 'Population': df_pop['Value']})
df_area = pd.DataFrame({'Area': df_area['Area'], 'Surface': df_area['Value']})

# Menambahkan baris yang hilang menggunakan pd.concat
missing_line = pd.DataFrame({'Area': ['Sudan'], 'Surface': [1886]})
df_area = pd.concat([df_area, missing_line], ignore_index=True)

# Menggabungkan tabel
d1 = pd.DataFrame(df.loc[:, ['Area', 'Item', 'Element']])
data = pd.merge(d1, df_pop, on='Area', how='left')
new_data = pd.merge(data, df_area, on='Area', how='left')

d3 = df.loc[:, 'Y1993':'Y2013'] 
data1 = new_data.join(d3)  

d4 = data1.loc[data1['Element'] == 'Food']  
d5 = d4.drop('Element', axis=1)
d5 = d5.fillna(0).infer_objects(copy=False) 

year_list = list(d3.iloc[:, :].columns)
d6 = d5.pivot_table(values=year_list, index=['Area'], aggfunc='sum')

italy = d4[d4['Area'] == 'Italy']
italy = italy.pivot_table(values=year_list, index=['Item'], aggfunc='sum')
italy = pd.DataFrame(italy.to_records())

item = d5.pivot_table(values=year_list, index=['Item'], aggfunc='sum')
item = pd.DataFrame(item.to_records())

d5 = d5.pivot_table(values=year_list, index=['Area', 'Population', 'Surface'], aggfunc='sum')
area = pd.DataFrame(d5.to_records())
d6.loc[:, 'Total'] = d6.sum(axis=1)
d6 = pd.DataFrame(d6.to_records())
d = pd.DataFrame({'Area': d6['Area'], 'Total': d6['Total'], 'Population': area['Population'], 'Surface': area['Surface']})









# RANKING

# Memproses data
year_list = list(df.iloc[:,10:].columns)
df_new = df.pivot_table(values=year_list, columns='Element', index=['Area'], aggfunc='sum')
df_fao = df_new.T

# Produsen hanya 'Food'
df_food = df_fao.xs('Food', level=1, axis=0)
df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()

# Produsen hanya 'Feed'
df_feed = df_fao.xs('Feed', level=1, axis=0)
df_feed_tot = df_feed.sum(axis=0).sort_values(ascending=False).head()

# Peringkat item yang paling banyak diproduksi
df_item = df.pivot_table(values=year_list, columns='Element', index=['Item'], aggfunc='sum')
df_item = df_item.T

# FOOD
df_food_item = df_item.xs('Food', level=1, axis=0)
df_food_item_tot = df_food_item.sum(axis=1).sort_values(ascending=False).head()  

# FEED
df_feed_item = df_item.xs('Feed', level=1, axis=0)
df_feed_item_tot = df_feed_item.sum(axis=1).sort_values(ascending=False).head() 

# Streamlit application
st.title('🥙 Food Data Analysis & Clustering')

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Cluster 1", "Cluster 2", "Cluster 3"],
    )

if selected == "Home":
    st.write('### Top 5 Food & Feed Producer')
    df_fao_tot = df_new.T.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_fao_tot)

    st.write('### Top 5 Food Producer')
    df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_food_tot)

    st.write('### Top 5 Feed Producer')
    df_feed_tot = df_feed.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_feed_tot)

    st.write('### Top 5 Food Produced Item')
    df_food_item_tot = df_food_item.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_food_item_tot)

    st.write('### Top 5 Feed Produced Item')
    df_feed_item_tot = df_feed_item.sum(axis=0).sort_values(ascending=False).head()
    st.bar_chart(df_feed_item_tot)










if selected == "Cluster 1":
    # CLUSTERING 1 - DBSCAN
    st.title('DBScan')

    # Mengambil data yang dibutuhkan untuk clustering
    X = pd.DataFrame({'Area': d['Area'], 'Total': d['Total'], 'Surface': d['Surface'], 'Population': d['Population']})

    # Input parameter untuk DBSCAN clustering menggunakan sidebar di streamlit
    st.sidebar.header('Choose Details for DBScan')
    eps = st.sidebar.number_input("Enter eps:", min_value=0.1, max_value=2.0, step=0.1, value=1.0)
    min_samples = st.sidebar.number_input("Enter min_samples:", min_value=1, max_value=10, step=1, value=2)

    # Memastikan data memiliki skala yang sama
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['Total', 'Surface', 'Population']])

    # Fungsi untuk melakukan DBSCAN clustering
    def DBSCAN_Clustering(X_scaled, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        return clusters

    # Fungsi untuk plot hasil clustering
    def Plot3dClustering(X, clusters):
        fig = px.scatter_3d(X, x='Total', y='Population', z='Surface', color=clusters,
                            labels={'Total': 'Total Production', 'Population': 'Population', 'Surface': 'Surface Area'},
                            title='3D Scatter plot of DBSCAN Clustering',
                            color_continuous_scale='Plasma', opacity=0.8)
        fig.update_layout(legend_title="Clusters")
        st.plotly_chart(fig)

    # Memanggil fungsi DBSCAN
    clusters = DBSCAN_Clustering(X_scaled, eps, min_samples)

    # Menambahkan cluster labels ke dataframe
    X['Cluster'] = clusters

    # Memanggil fungsi untuk plot DBSCAN
    st.write('### 3D Scatter Plot of DBSCAN Clustering')
    Plot3dClustering(X, clusters)

    # Menampilkan informasi untuk tiap cluster
    st.subheader('Cluster Details')
    unique_labels = np.unique(clusters)
    for label in unique_labels:
        if label == -1:
            st.write("Noise:")
            cluster_members = X[X['Cluster'] == label]
            if not cluster_members.empty:
                best_area = cluster_members.loc[
                    cluster_members[['Surface', 'Population', 'Total']].sum(axis=1).idxmax()
                ]

                #Mencari data yang paling tinggi secara luas area, populasi, dan total produksi dalam noise
                st.write(cluster_members[['Area', 'Total', 'Surface', 'Population']])
                st.write(f"\nThe biggest area with most population and most production in Noise is {best_area['Area']} with area {best_area['Surface']}, population {best_area['Population']}, and production {best_area['Total']}.\n")
        else:
            st.write(f"Cluster {label + 1}:")
            cluster_members = X[X['Cluster'] == label]
            if not cluster_members.empty:
                best_area = cluster_members.loc[
                    cluster_members[['Surface', 'Population', 'Total']].sum(axis=1).idxmax()
                ]

                #Mencari data yang paling tinggi secara luas area, populasi, dan total produksi dalam cluster
                st.write(cluster_members[['Area', 'Total', 'Surface', 'Population']])
                st.write(f"\nThe biggest area with most population and most production in Cluster {label + 1} is {best_area['Area']} with area {best_area['Surface']}, population {best_area['Population']}, and production {best_area['Total']}.\n")

        st.write("\n")








    # CLUSTERING 1 - KMEANS
    st.title('K-Means')
    # Menetapkan judul aplikasi Streamlit dan input jumlah cluster yang diinginkan
    st.sidebar.header('Choose Details for K-Means')
    num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=1, max_value=10, step=1)

    # Melakukan metode Clustering K-Means pada dataset yang sudah disediakan
    def K_Means(X, n):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=7, random_state=0)
        model.fit(X_scaled)
        clust_labels = model.predict(X_scaled)
        cent = model.cluster_centers_
        return clust_labels, cent

    # Membuat scatterplot 3D secara interaktif dari hasil clustering menggunakan Plotly
    def Plot3dClusteringKMeans(X, clusters):
        fig = px.scatter_3d(X, x='Total', y='Population', z='Surface', color=clusters,
                            labels={'Total': 'Total Production', 'Population': 'Population', 'Surface': 'Surface Area'},
                            title=f'3D Clustering with {num_clusters} clusters (K-Means)',
                            color_continuous_scale='Plasma', opacity=0.8)
        fig.update_layout(legend_title="Clusters")
        st.plotly_chart(fig)

    # Preprocessing: memastikan bahwa hanya data numerik yang digunakan untuk clustering
    X_numeric = X[['Total', 'Surface', 'Population']].copy()

    # Menghitung WCSS untuk berbagai jumlah cluster menggunakan metode Elbow
    wcss = []
    for i in range(1, 8):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=7, random_state=0)
        kmeans.fit(X_numeric)
        wcss.append(kmeans.inertia_)

    # Melakukan Clustering K-Means dan ditambahkan ke dalam dataframe
    clust_labels_kmeans, cent = K_Means(X_numeric, num_clusters)
    X['KMeans_Cluster'] = clust_labels_kmeans

    # Menampilkan hasil grafik Metode Elbow
    st.write('### Elbow Method for Optimal K in K-Means')
    # Plot Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 8), wcss, marker='o')
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS') 
    st.pyplot(fig)

    # Menampilkan 3D scatterplot dari hasil clustering
    st.write(f'### 3D Scatter Plot of K-Means Clustering with {num_clusters} clusters')
    Plot3dClusteringKMeans(X, 'KMeans_Cluster')

    # Memberikan informasi rinci tentang setiap cluster
    st.subheader('Cluster Details (K-Means)')
    for i in range(num_clusters):
        st.write(f"Cluster {i}:\n")
        cluster_kmeans = X[X['KMeans_Cluster'] == i][['Area', 'Total', 'Population', 'Surface']]
        st.dataframe(cluster_kmeans)
        st.write("\n")

        # Menyoroti area terbaik dalam setiap cluster
        best_area = cluster_kmeans.loc[cluster_kmeans['Total'].idxmax()]
        st.write(f"The best area to produce in Cluster {i} is {best_area['Area']} with a total production of {best_area['Total']:.1f}.\n")

    








if selected == "Cluster 2":
    # CLUSTERING 2 - DBSCAN
    st.title('Clustering Production of Food Items with DBScan')

    # User input untuk nama area yang akan di clustering
    area_name = st.sidebar.text_input("Enter the area name for clustering:")

    # Input parameter untuk DBSCAN clustering menggunakan sidebar di streamlit
    st.sidebar.header('Choose Details for DBScan')
    eps = st.sidebar.number_input("Enter eps:", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
    min_samples = st.sidebar.number_input("Enter min_samples:", min_value=1, max_value=10, step=1, value=2)

    if area_name:
        # Memanggil fungsi 'prepare_data_for_area' dengan parameter 'area_name'
        area_total = prepare_data_for_area(area_name)

        # Menampilkan total produksi per item dari tahun 1993 hingga 2013 di area yang dipilih
        st.write(f"Total produksi per item dari tahun 1993 hingga 2013 di {area_name}:")
        st.write(area_total[['Item', 'Total']])

        # Menyiapkan data yang akan di clustering
        Y = pd.DataFrame({'Item': area_total['Item'], 'Total': area_total['Total']})

        # Meng-encode kolom item menjadi kolom Item_encoded dalam bentuk numerik
        label_encoder = LabelEncoder()
        Y['Item_encoded'] = label_encoder.fit_transform(Y['Item'])

        # Mengambil kolom Total
        Y_scaled = Y[['Total']]

        # Melakukan standardisasi pada kolom Total
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y_scaled)

        # Menjalankan DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
        clusters = dbscan.fit_predict(Y_scaled)

        # Menambahkan cluster ke dalam DataFrame Y
        Y['Cluster'] = clusters

        # Menyimpan item berdasarkan cluster
        clustered_items = {}
        for cluster in np.unique(clusters):
            items_in_cluster = Y[Y['Cluster'] == cluster][['Item', 'Total']]
            if cluster == -1:
                cluster_label = 'Noise'
            else:
                cluster_label = f'Cluster {cluster + 1}'
            clustered_items[cluster_label] = items_in_cluster

        # Menampilkan hasil clustering
        st.write("Hasil Clustering:")
        for cluster_label, items in clustered_items.items():
            st.write(f"\n{cluster_label}:")
            st.write(items)

        # Menemukan item terbaik untuk diproduksi di setiap cluster
        st.write(f"#### Conclusion for each cluster")
        for cluster_label, items in clustered_items.items():
            best_item = items.loc[items['Total'].idxmax()]
            st.write(f"The best item to produce in {cluster_label} is {best_item['Item']} with a total production of {best_item['Total']}.")

        # Visualisasi hasil clustering dalam scatter plot dengan warna berbeda untuk setiap cluster
        unique_clusters = np.unique(clusters) 
        colors = px.colors.qualitative.Plotly

        # Membuat figure untuk plot
        fig = go.Figure()
        for i, cluster in enumerate(unique_clusters):
            cluster_data = Y[Y['Cluster'] == cluster]
            color = 'black' if cluster == -1 else colors[i % len(colors)] 
            label = 'Noise' if cluster == -1 else f'Cluster {cluster + 1}' 
            fig.add_trace(go.Scatter(
                x=cluster_data['Item_encoded'],
                y=cluster_data['Total'],
                mode='markers',
                marker=dict(color=color),
                name=label,
                text=[f'Item: {item}<br>Total: {total}' for item, total in zip(cluster_data['Item'], cluster_data['Total'])],
                hoverinfo='text'
            ))

        fig.update_layout(
            title=f'Clustered Production of Food Items in {area_name}',
            xaxis_title='Item',
            yaxis_title='Total Production',
            xaxis=dict(
                tickmode='array',
                tickvals=Y['Item_encoded'],
                ticktext=Y['Item']
            ),
            width=1000,
            height=500
        )

        # Menampilkan plot di streamlit
        st.plotly_chart(fig)








    # CLUSTERING 2 - KMEANS
    # Menetapkan judul aplikasi Streamlit dan input jumlah cluster yang diinginkan
    st.title('Clustering Production of Food Items with K-Means')

    # Menerima input nama area dan jumlah cluster
    st.sidebar.header('Choose Details for K-Means')
    num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=1, value=3)

    # Menentukan jumlah cluster yang optimal dengan menggunakan Metode Elbow
    def elbow_method(data, max_clusters=10):
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
    
        # Menampilkan hasil visualisasi Metode Elbow
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, max_clusters + 1), wcss, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')  
        st.pyplot(fig)

    if area_name:
        # Menyiapkan data dengan memanggil fungsi
        area_total = prepare_data_for_area(area_name)

        # Menampilkan total produksi per item dari tahun 1993 hingga 2013 di area yang dipilih
        st.write(f"Total produksi per item dari tahun 1993 hingga 2013 di {area_name}:")
        st.write(area_total[['Item', 'Total']])

        # Menyiapkan data pada dataframe
        Y = pd.DataFrame({'Item': area_total['Item'], 'Total': area_total['Total']})

        # Mengonversi kategori ke dalam sifat numerik
        label_encoder = LabelEncoder()
        Y['Item_encoded'] = label_encoder.fit_transform(Y['Item'])

        Y_scaled = Y[['Total']]

        # Melakukan normalisasi data "Total"
        scaler = StandardScaler()
        Y_scaled = scaler.fit_transform(Y_scaled)

        # Menggunakan Metode Elbow untuk membantu menentukan jumlah cluster yang optimal
        elbow_method(Y_scaled, max_clusters=10)

        # Menampilkan hasil clustering menggunakan Metode K-Means
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        clusters = kmeans.fit_predict(Y_scaled)

        # Tambah label dan hasil clustering pada dataframe
        Y['Cluster'] = clusters
        area_total['Cluster'] = clusters

        # Menampilkan item dalam setiap cluster
        clustered_items = {}
        for cluster in np.unique(clusters):
            items_in_cluster = Y[Y['Cluster'] == cluster][['Item', 'Total']]
            cluster_label = f'Cluster {cluster + 1}'
            clustered_items[cluster_label] = items_in_cluster

        for cluster_label, items in clustered_items.items():
            st.write(f"\n{cluster_label}:")
            st.write(items)

        # Menentukan item terbaik di setiap cluster
        st.write(f"#### Conclusion for each cluster")
        for cluster_label, items in clustered_items.items():
            best_item = items.loc[items['Total'].idxmax()]
            st.write(f"The best item to produce in {cluster_label} is {best_item['Item']} with a total production of {best_item['Total']}.")

        # Membuat plot
        unique_clusters = np.unique(clusters)
        colors = px.colors.qualitative.Plotly

        fig = go.Figure()
        for i, cluster in enumerate(unique_clusters):
            cluster_data = Y[Y['Cluster'] == cluster]
            color = 'black' if cluster == -1 else colors[i % len(colors)]
            label = f'Cluster {cluster + 1}'
            fig.add_trace(go.Scatter(
                x=cluster_data['Item_encoded'],
                y=cluster_data['Total'],
                mode='markers',
                marker=dict(color=color),
                name=label, 
                text=[f'Item: {item}<br>Total: {total}' for item, total in zip(cluster_data['Item'], cluster_data['Total'])],  # Add hover text
                hoverinfo='text'
            ))

        fig.update_layout(
            title=f'Clustered Production of Food Items in {area_name}',
            xaxis_title='Item',
            yaxis_title='Total Production',
            xaxis=dict(
                tickmode='array',
                tickvals=Y['Item_encoded'],
                ticktext=Y['Item']
            ),
            width=1200, 
            height=600  
        )
        
        st.plotly_chart(fig)









if selected == "Cluster 3":
    st.title('Clustering of Agricultural Areas Based on Production Data')

    # CLUSTERING 3 - DBSCAN
    st.write("## DBScan")

    # Memilih kolom yang diperlukan
    production_columns = ['Y' + str(year) for year in range(1993, 2014)]
    selected_columns = ['Area', 'Item', 'Element', 'latitude', 'longitude'] + production_columns
    production_data = df[selected_columns].copy()
        
    # Sidebar untuk pemilihan item
    items = production_data['Item'].unique()
    item_type = st.sidebar.selectbox("Select item type for clustering", items)
    st.sidebar.header('Choose Details for DBScan')

    # Parameter DBSCAN
    eps = st.sidebar.slider("Enter eps value", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
    min_samples = st.sidebar.slider("Enter min_samples value", min_value=1, max_value=10, step=1, value=2)
        
    # Mengambil subset data untuk item terpilih dan elemen 'Food'
    subset_data = production_data[(production_data['Item'] == item_type) & (production_data['Element'] == 'Food')].copy()
        
    # Menghitung total produksi dari tahun 1993 hingga 2013
    subset_data['total_production_1993_2013'] = subset_data[production_columns].sum(axis=1)
        
    # Memilih fitur untuk clustering
    X = subset_data[['latitude', 'longitude', 'total_production_1993_2013']].values
        
    # Standarisasi fitur menggunakan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    # Clustering dengan DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
        
    # Menambahkan label cluster ke data subset
    subset_data['cluster_label'] = clusters
        
    # Memisahkan data yang tercluster dan poin noise
    clustered = subset_data[subset_data['cluster_label'] != -1]
    noise = subset_data[subset_data['cluster_label'] == -1]
    unique_labels = np.unique(clusters)

    # Plot hasil clustering dalam plot 3D scatter
    fig = px.scatter_3d(clustered, x='longitude', y='latitude', z='total_production_1993_2013',
                        color='cluster_label', opacity=0.8, size_max=15,
                        title=f'3D Scatter Plot with DBSCAN Clustering for {item_type}',
                        color_continuous_scale='Plasma')
    # Menambahkan poin noise ke plot
    fig.add_trace(px.scatter_3d(noise, x='longitude', y='latitude', z='total_production_1993_2013',
                                        color_discrete_sequence=['black'], symbol='cluster_label',
                                        opacity=0.8, size_max=15).data[0])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    # Menampilkan plot di Streamlit
    st.plotly_chart(fig)

    # Mencari area terbaik di setiap cluster
    def find_best_areas(clustered_data):
        unique_labels = clustered_data['cluster_label'].unique()
        results = []
    
        for label in unique_labels:
            if label != -1:
                cluster_data = clustered_data[clustered_data['cluster_label'] == label]
                max_production = cluster_data['total_production_1993_2013'].max()
                best_area = cluster_data[cluster_data['total_production_1993_2013'] == max_production].iloc[0]
                result = {
                    'cluster_label': label,
                    'best_area': best_area['Area'],
                    'total_production': best_area['total_production_1993_2013']
                }
                results.append(result)
    
        return results
        
    # Menampilkan data poin yang termasuk dalam noise dan setiap cluster
    for label in unique_labels:
        if label == -1:
            st.write("### Data Points in Noise:")
            st.write(noise[['Area', 'latitude', 'longitude', 'total_production_1993_2013']])
            
            if not noise.empty:
                max_production_noise = noise['total_production_1993_2013'].max()
                best_area_noise = noise[noise['total_production_1993_2013'] == max_production_noise].iloc[0]
                st.write(f"The best area to produce in Noise is {best_area_noise['Area']} with a total production of {best_area_noise['total_production_1993_2013']}.")
        else:
            st.write(f"### Data Points in Cluster {label}:")
            cluster_data = clustered[clustered['cluster_label'] == label]
            st.write(cluster_data[['Area', 'latitude', 'longitude', 'total_production_1993_2013']])

            results = find_best_areas(cluster_data)

            for result in results:
                st.write(f"The best area to produce in Cluster {result['cluster_label']} is {result['best_area']} with a total production of {result['total_production']}.")







    # CLUSTER 3 KMEANS
    st.write("## K-Means")

    # Input dari user
    st.sidebar.header('Choose Details for K-Means')
    num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=2, step=1)

    production_columns = ['Y' + str(year) for year in range(1993, 2014)]
    selected_columns = ['Area', 'Item', 'Element', 'latitude', 'longitude'] + production_columns

    # Membuat DataFrame baru dengan kolom yang dipilih
    production_data = df[selected_columns].copy()

     # Menambahkan kolom total produksi selama tahun 1993-2013
    production_data['total_production_1993_2013'] = production_data[production_columns].sum(axis=1)
    
    production_data.fillna(0, inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(production_data[production_columns + ['latitude', 'longitude']])

    # Elbow Method
    def elbow_method(df, max_clusters=10):
        inertia = []
        for k in range(1, min(max_clusters + 1, len(df) + 1)):  # Sesuaikan max_clusters dengan jumlah data
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(df)
            inertia.append(kmeans.inertia_)
        return inertia

    # Melakukan clustering berdasarkan jumlah cluster, jenis item, latitude, dan longitude
    def perform_clustering(num_clusters, item_type):
        item_type = item_type.lower()

        item_data = production_data[(production_data['Item'].str.lower() == item_type) &
                                (production_data['Element'].str.lower() == 'food')].copy()

        scaled_item_data = scaler.fit_transform(item_data[production_columns + ['latitude', 'longitude']])

        inertia = elbow_method(scaled_item_data)

        # Memilih jumlah cluster berdasarkan Elbow Method
        st.subheader('Elbow Method Plot')
        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal K')
        st.pyplot(fig)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        item_data['cluster'] = kmeans.fit_predict(scaled_item_data)

        st.subheader('3D Scatter Plot')
        fig = px.scatter_3d(
            item_data, 
            x='longitude', 
            y='latitude', 
            z='total_production_1993_2013', 
            color='cluster', opacity=0.8, 
            title=f'Clustering of Areas Based on Production Data for "{item_type}"',
            labels={
                'longitude': 'Longitude',
                'latitude': 'Latitude',
                'total_production_1993_2013': 'Total Production 1993-2013',
                'cluster': 'Cluster'
            },
            color_continuous_scale='Viridis'
        )
    
        st.plotly_chart(fig)

        # Menampilkan hasil clustering per cluster dalam bentuk tabel
        cluster_tables = []
        best_areas = []
        for cluster_id in range(num_clusters):
            cluster_data = item_data[item_data['cluster'] == cluster_id]
            cluster_table = cluster_data[['Area', 'latitude', 'longitude', 'total_production_1993_2013']]
            cluster_tables.append(cluster_table)

            best_area = cluster_table.loc[cluster_table['total_production_1993_2013'].idxmax()]
            best_areas.append(best_area)

        for i, (table, best_area) in enumerate(zip(cluster_tables, best_areas), start=1): 
            st.subheader(f'Cluster {i}:')
            st.write(table)
            st.write(f"The best area to produce in Cluster {i} is {best_area['Area']} with a total production of {best_area['total_production_1993_2013']:.1f}.\n")

    perform_clustering(num_clusters, item_type)
