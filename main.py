import networkx as nx
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import os
from optparse import OptionParser
from scipy.stats import multinomial

import htest
def clusteringProf(points):
    hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(points)
    return y_hc

def mapClusterTweets(labels,messages):
    annotations = []
    for label in labels:
        tweet = {}
        tweet[0] = label
        tweet[1] = messages[label]
        for i in range(2, len(labels[label])+2):
            tweet[i] = labels[label][i-2]
        annotations.append(tweet)
    df = pd.DataFrame(annotations)
    points = df.iloc[:, 3:]

    for col in points:
        points[col] = points[col].fillna(0)
    clusters = clusteringProf(points)
    data_with_clusters = df.copy()
    data_with_clusters['Clusters'] = clusters

    data_with_clusters.to_excel("final_cluster.xlsx", sheet_name='output')



def analyzeGraph(graph,isClass):
    print('Total number of nodes: ', graph.number_of_nodes())
    print('Total number of edges: ', graph.number_of_edges())
    print('Number of Connected Components: ', nx.number_connected_components(graph))

    cc_index = 1
    for cc in nx.connected_components(graph):
        print(f'Connected Component {cc_index} with size: {len(cc)}')
        cc_index += 1

    print('Density of the graph: ', nx.density(graph))
    print('Average Degree: ', graph.number_of_edges() / graph.number_of_nodes())
    print('Is the graph directed?: ', nx.is_directed(graph))

    clustering_coefficient = nx.algorithms.cluster.clustering(graph)
    avg_clustering_coefficient = sum(clustering_coefficient.values()) / len(clustering_coefficient)
    print('Average Clustering Coefficient: ', avg_clustering_coefficient)

    if isClass :
        name = "class"
    else:
        name = "prof"
    plt.hist(nx.degree_histogram(graph))
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.title("Histogram - Degree")
    plt.savefig("Histogram_Degree_"+name+".png")
    plt.close()

    # Plotting the histogram of size of connected components
    plt.hist([len(c) for c in nx.connected_components(graph)])
    plt.xlabel("Size (Number of Nodes)")
    plt.ylabel("Count")
    plt.title("Histogram - Size of connected components")
    plt.savefig("Histogram_Size_ConnectedComponents_"+name+".png")
    plt.close()

    # mylist = []
    # for n in nx.nodes(graph):
    #     points = n.replace("(", "").replace(")", "").split(",")
    #     mylist.append(list(map(float, points)))
    # clusters = clusteringProf(pd.DataFrame(mylist))
    # print("len ",len(clusters))
    # graph = coloredGraph(graph, clusters)

    # nx.write_gexf(graph,  name+"_colorGraph.gexf")
    #
    # return clusters



def coloredGraph(graph, clusters):
    i = 0
    for n in graph.nodes():
        graph.nodes[n]["color"] = clusters[i]
        i+=1
    return graph


def createCommonFile():
    dictOur = {}
    Message = {}
    for filename in os.listdir("Project 2 annotated data"):
        if filename.endswith(".xlsx"):
            sheets = pd.read_excel(open("Project 2 annotated data/"+filename, 'rb'), sheet_name='Data to Annotate')
            for c in sheets:
                sheets[c] = sheets[c].fillna(0)
            for index, row in sheets.iterrows():
                messageId = row[0]
                features = row[2:].values.tolist()
                for i in range(0, len(features)):
                    if (isinstance(features[i], str)):
                        features[i] = 0
                if messageId in dictOur:
                    for i in range(0, len(dictOur[messageId])):
                        dictOur[messageId][i] = dictOur[messageId][i] + features[i]
                else:
                    dictOur[messageId] = features
                    Message[messageId] = row[1]
    return (dictOur, Message)

def graphnPldl(Y):
    optparser = OptionParser()
    optparser.add_option('-s', '--sample',
                         dest='sample_size',
                         help='Estimated sample size of each input',
                         default=10,
                         type='float')
    optparser.add_option('-c', '--confidence',
                         dest='confidence',
                         help='Confidence (float) of regions desired',
                         default=0.9,
                         type='float')

    (options, args) = optparser.parse_args()

    t = {}
    for x, y in Y.items():
        if (sum(y) == 0):
            break
        y1 = multinomial(options.sample_size, [yi / sum(y) for yi in y])
        y2 = htest.most_likely(y1)
        t[tuple(y2)] = x

    friendlist = []

    for x, y in Y.items():
        if (sum(y) == 0):
            break
        my = multinomial(options.sample_size, [yi / sum(y) for yi in y])
        mcr = htest.min_conf_reg(my, options.confidence)
        ldls = [tuple(i) for i in mcr]

        friends = set(ldls) & set(t.keys())
        for friend in friends:
            if (tuple(y) != friend):
                friendlist.append((tuple(y), friend))

    g = nx.Graph(friendlist)
    newFile = f"class_common_{options.confidence}_{options.sample_size}.gexf"
    nx.write_gexf(g, newFile)
    return newFile


def main():
    graph = nx.read_gexf("label_space_0.9_10.gexf")
    analyzeGraph(graph,False)
    (labels,Message) = createCommonFile()
    graph_class = graphnPldl(labels)
    graph = nx.read_gexf(graph_class)
    analyzeGraph(graph,True)
    mapClusterTweets(labels,Message)

if __name__ == '__main__':
    main()