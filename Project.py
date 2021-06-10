import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats

# df=pd.read_csv('close_prices.csv')
# df.drop(columns=['date'],inplace=True)
# df=df.sample(frac=1)
# df.reset_index(inplace=True)
# df.drop(columns=['index'],inplace=True)
# df.to_csv('data.csv',index_label='index')
df = pd.read_csv ( 'data.csv', index_col='index' )
print(df.head())


def info(df):
    """
    :param df-pd.DataFrame
    :return-info(Данные записываются в excel(info.xlsx):
    Статистическая информация о датасете
    """
    info = df.describe ()
    info.to_excel ( 'info.xlsx' )
    return info


def preprocessing(df):
    """
    :param df-pd.Dataframe:
    :return - df-pd.Dataframe:
    Иллюстрация ящика с усами для определения выбросов
    удаление всех значений, которые отклонились от своего среднего значения более чем на 3 сигмы(Правило 3-сигм)
    """
    sns.boxplot ( data=df.loc [ :, : ], orient='h' )
    plt.show ()
    missing_values_count = df.isnull ().sum ()
    print ( missing_values_count )
    for i in df.columns:
        df = df [ np.abs ( df [ i ] - df [ i ].mean () ) <= (3 * df [ i ].std ()) ]
    return df


def correlation_pearson(df, threshold):
    """
    Таблица парных коэффициентов и их значимостей между каждый "х" и "y"
    :param df-pd.Dataframe:
    :param threshold-порог корреляции:
    :return-top_corr_pearson(Данные записываются в excel(correlation matrix pearson.xlsx)
                (Данные записываются в excel('top_correlation_pearson.xlsx')
    """
    corr = df.corr ()
    values = df.corr ( method=lambda x, y: scipy.stats.pearsonr ( x, y ) [ 1 ] ) - np.eye ( len ( df.columns ) )
    for m, n in zip ( *(corr.values, values.values) ):
        for i, j in zip ( m, n ):
            t = 'coef' + ' ' + str ( round(i,2) ) + ' ' + 'p-value' + ' ' + str ( round(j,2))
            corr.replace ( i, t, inplace=True )
    corr.to_excel ( 'correlation matrix pearson.xlsx' )

    def get_redundant_pairs(df):
        """
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        :param df-pd.Dataframe:
        :return - pairs_to_drop(удаление повторяющихся значений в корреляционной таблице:
        """
        pairs_to_drop = set ()
        cols = df.columns
        for i in range ( 0, df.shape [ 1 ] ):
            for j in range ( 0, i + 1 ):
                pairs_to_drop.add ( (cols [ i ], cols [ j ]) )
        return pairs_to_drop

    def get_top_abs_correlations(df, threshold):
        """
        Отбор коэффциентов корреляций превышающих порог threshold
        :param df-pd.Dataframe:
        :param threshold:
        :return - top_correlation(Значение корреляции>threshold:
        """
        au_corr = df.corr ().abs ().unstack ()
        labels_to_drop = get_redundant_pairs ( df )
        au_corr = au_corr.drop ( labels=labels_to_drop ).sort_values ( ascending=False )
        for i in range ( len ( au_corr ) ):
            if au_corr [ i ] > threshold:
                top_correlation = au_corr [ 0:round(i,2) ]
        return top_correlation

    top_corr_pearson = get_top_abs_correlations ( df, threshold )
    top_corr_pearson.to_excel ( 'top_correlation_pearson.xlsx' )
    return top_corr_pearson


def correlation_spearman(df, threshold):
    """
    Таблица парных коэффициентов и их значимостей между каждый "х" и "y"
    :param df-pd.Dataframe:
    :param threshold-порог корреляции:
    :return-top_corr_spearman(Данные записываются в excel(correlation matrix spearman.xlsx)
                (Данные записываются в excel('top_correlation_spearman.xlsx')
    """
    corr = df.corr ( method='spearman' )
    values = df.corr ( method=lambda x, y: scipy.stats.spearmanr ( x, y ) [ 1 ] ) - np.eye ( len ( df.columns ) )
    for m, n in zip ( *(corr.values, values.values) ):
        for i, j in zip ( m, n ):
            t = 'coef' + ' ' + str ( i ) + ' ' + 'p-value' + ' ' + str ( j )
            corr.replace ( i, t, inplace=True )
    corr.to_excel ( 'correlation matrix spearman.xlsx' )

    def get_redundant_pairs(df):
        """
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        :param df-pd.Dataframe:
        :return - pairs_to_drop(удаление повторяющихся значений в корреляционной таблице:
        """
        pairs_to_drop = set ()
        cols = df.columns
        for i in range ( 0, df.shape [ 1 ] ):
            for j in range ( 0, i + 1 ):
                pairs_to_drop.add ( (cols [ i ], cols [ j ]) )
        return pairs_to_drop

    def get_top_abs_correlations(df, threshold):
        """
        Отбор коэффциентов корреляций превышающих порог threshold
        :param df-pd.Dataframe:
        :param threshold:
        :return - top_correlation(Значение корреляции>threshold:
        """
        au_corr = df.corr ( method='spearman' ).abs ().unstack ()
        labels_to_drop = get_redundant_pairs ( df )
        au_corr = au_corr.drop ( labels=labels_to_drop ).sort_values ( ascending=False )
        for i in range ( len ( au_corr ) ):
            if au_corr [ i ] > threshold:
                top_correlation = au_corr [ 0:i ]
        return top_correlation

    top_corr_spearman = get_top_abs_correlations ( df, threshold )
    top_corr_spearman.to_excel ( 'top_correlation_spearman.xlsx' )
    return top_corr_spearman


def dimensionality_reduction_pca(df):
    """
    Использование методики Главных компонент для уменьшения размерности данных
    :param df-pd.Dataframe:
    :return data:
    """
    x = df.values
    x = StandardScaler ().fit_transform ( x )
    pca = PCA ( n_components=0.90)
    pca.fit ( x )
    data = pd.DataFrame ( data=pca.fit_transform ( x ),
                          columns=[ 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6'] )
    print ( pca.explained_variance_ratio_ )
    for i in range ( pca.n_components_ ):
        print ( df.columns [ pca.components_ [ i ].argsort () [ -3: ] [ ::-1 ] ] )
    return data


def dimensionality_reduction_fca(df):
    """
    Использование методики Факторного анализа для нахождения латентных переменных
    :param df-pd.Dataframe:
    :return data:
    """
    X = df.values
    X = StandardScaler ().fit_transform ( X )
    fa = FactorAnalyzer ()
    fa.fit ( X )
    ev, v = fa.get_eigenvalues ()
    sns.lineplot ( x=range ( 1, X.shape [ 1 ] + 1 ), y=ev )
    plt.show ()
    fa = FactorAnalyzer ( 3, rotation='varimax' )
    fa.fit ( X )
    data = fa.transform ( X )
    data = pd.DataFrame ( data=data, columns=[ 'f_1', 'f_2', 'f_3' ] )
    return data


def dimensionality_reduction_mds(df,n):
    """
    Использование методики Многомерного шкалирования для возможного иллюстрирования данных в размерности 2|3
    :param df-pd.Dataframe:
    :return data:
    """
    x = df.values
    scaler = MinMaxScaler ()
    x_scaled = scaler.fit_transform ( x )
    mds = MDS ( n, random_state=0 )
    x_new = mds.fit_transform ( x_scaled )
    data = pd.DataFrame ( data=x_new, columns=[ 'x', 'y'] )
    return data


def dimensionality_reduction_t_sne(df):
    """
    Использование методики Стохастическое вложение соседей с t-распределением для возможного иллюстрирования данных в размерности 2|3
    :param df-pd.Dataframe:
    :return data:
    """
    x = df.values
    tsne = TSNE ( n_components=2, verbose=1, perplexity=20, n_iter=300 )
    x_new = tsne.fit_transform ( x )
    return x_new


def elbow_method(df):
    """
    Использование эвристики "метод колен" для определения количества кластера в наборе данных
    :param df-pd.Dataframe:
    :pass(Графическая интерпретация):
    """
    inertia = [ ]
    for i in range ( 1, 11 ):
        kmeans = KMeans ( n_clusters=i, random_state=42 )
        kmeans.fit ( df )
        inertia.append ( kmeans.inertia_ )
    sns.lineplot ( range ( 1, 11 ), inertia, )
    plt.xlabel ( 'Number of clusters' )
    plt.ylabel ( 'WCSS' )
    plt.show ()
    return


def clustering_kmeans(df,n):
    """
    Использование алгоритма K-Means для кластеризации данных
    :param df-pd.Dataframe:
    :param n_clusters-(количество кластеров):
    :return data:
    """
    labels = pd.Series( KMeans ( n_clusters=n, random_state=42, init='k-means++' ).fit ( df ).labels_,
                         name='label' )
    data = pd.concat ( [ df, labels ], axis=1 )
    data.to_csv ( 'data2_labeled.csv', index_label='index' )
    return labels


def visualizing_3d(data):
    """
    Visualizing data into 3-dimensional graph
    :param data:
    :pass(Visualization):
    """
    fig = plt.figure ( figsize=(10, 10) )
    ax = Axes3D ( fig )
    g = ax.scatter ( data.x, data.y, data.z, c=data.label )
    ax.set_xlabel ( 'X Label' )
    ax.set_ylabel ( 'Y Label' )
    ax.set_zlabel ( 'Z Label' )
    legend = ax.legend ( *g.legend_elements () )
    ax.add_artist ( legend )
    plt.show ()
    pass


def visualizing_2d(data):
    """
    Visualizing data into 2-dimensional graph
    :param data:
    :pass(Visualization):
    """
    sns.scatterplot ( data=data, x='x', y='y', hue='label',palette='deep')
    plt.show ()
    pass
info(df)
print(len(df))
cleaned_df=preprocessing(df)
correlation_pearson(cleaned_df,0.9)
print(len(cleaned_df))
elbow_method(cleaned_df)
labels=clustering_kmeans(cleaned_df,4)
reduced_data_pca=dimensionality_reduction_pca(cleaned_df)
reduced_data_mds=pd.DataFrame(data=dimensionality_reduction_mds(reduced_data_pca,2))
print(df.shape)
print(labels.shape)
print(reduced_data_pca.shape)
print(reduced_data_mds.shape)
data=pd.concat([reduced_data_mds,labels],axis=1)
print(data.head())
visualizing_2d(data)