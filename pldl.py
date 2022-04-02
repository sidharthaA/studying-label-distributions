from collections import Counter
import networkx as nx
from optparse import OptionParser
import pandas as pd
from scipy.stats import multinomial
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score

import htest

optparser = OptionParser()
optparser.add_option('-f', '--inputFile',
                     dest='input_file',
                     help='json input filename',
                     default="jobQ3_BOTH_train.json")
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

df = pd.read_json(options.input_file, orient='split')
Y_dict = (df.groupby('message_id')
    .apply(lambda x: dict(zip(x['worker_id'],x['label_vector'])))
    .to_dict())
Ys = {x: list(y.values()) for x,y in Y_dict.items()}
Yz = {x: Counter(y) for x,y in Ys.items()}
dims = max([max(y.values()) for x,y in Yz.items()])+1
Y = {x:[Yz[x][i] if i in Yz[x] else 0 for i in range(dims)] for x,y in Yz.items()}
labels = df.groupby(['label', 'label_vector']).first().index.tolist()
Yframe = pd.DataFrame.from_dict(Y, orient='index')
XnY = df.groupby("message_id").first().join(Yframe, on="message_id")[['message',0,1,2,3,4,5,6,7,8,9,10,11]]


t = {}
for x,y in Y.items():
    y1 = multinomial(options.sample_size, [yi/sum(y) for yi in y])
    y2 = htest.most_likely(y1)
    t[tuple(y2)] = x

friendlist = []


for x,y in Y.items():
    print (f"x: {x}, y: {y}")
    #my = multinomial(sum(y), [yi/sum(y) for yi in y])
    my = multinomial(options.sample_size, [yi/sum(y) for yi in y])
    mcr = htest.min_conf_reg(my, options.confidence)
    #ldls = [[int(i) for i in m.p * m.n] for m in mcr]
    ldls = [tuple(i) for i in mcr]
    friends = []
    """
    for mc in mcr:
        if tuple(mc) in t:
            friends.append(mc)
    """
    friends = set(ldls) & set(t.keys())
    for friend in friends:
        if (tuple(y) != friend):
            friendlist.append((tuple(y),friend))

g = nx.Graph(friendlist)
nx.write_gexf(g,f"label_space_{options.confidence}_{options.sample_size}.gexf")

# Analysis on the Graph
print('Total number of nodes: ', g.number_of_nodes())
print('Total number of edges: ', g.number_of_edges())
print('Number of Connected Components: ', nx.number_connected_components(g))

cc_index = 1
for cc in nx.connected_components(g):
    print(f'Connected Component {cc_index} with size: {len(cc)}')
    cc_index += 1

print('Density of the graph: ', nx.density(g))
print('Average Degree: ', g.number_of_edges()/g.number_of_nodes())
print('Is the graph directed?: ', nx.is_directed(g))

clustering_coefficient = nx.algorithms.cluster.clustering(g)
avg_clustering_coefficient = sum(clustering_coefficient.values())/len(clustering_coefficient)
print('Average Clustering Coefficient: ', avg_clustering_coefficient)

# Plotting the histogram of degree
plt.hist(nx.degree_histogram(g))
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.title("Histogram - Degree")
plt.savefig("Histogram_Degree.png")
plt.close()

# Plotting the histogram of size of connected components
plt.hist([len(c) for c in nx.connected_components(g)])
plt.xlabel("Size (Number of Nodes)")
plt.ylabel("Count")
plt.title("Histogram - Size of connected components")
plt.savefig("Histogram_Size_ConnectedComponents.png")
plt.close()

# Computing the Cohen's kappa interannotator
df_1 = pd.read_excel("JobQ3a_si9808.xlsx", sheet_name='Data to Annotate', engine='openpyxl')

df_2 = pd.read_excel("JobQ3a_as9125.xlsx", sheet_name='Data to Annotate', engine='openpyxl')

for label_no in range(2, 14):
    y_1 = df_1[df_1.columns[label_no]].fillna(0).astype(int)
    y_2 = df_2[df_2.columns[label_no]].fillna(0).astype(int)
    confusion_mat = confusion_matrix(y_1, y_2)
    labels = [0, 1]
    displayCM = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
    displayCM.plot()
    title = 'Label Number ' + str(label_no-1)
    plt.title(title)
    fig_name = title.replace(" ", '_')
    plt.savefig(fig_name + ".png")
    print(f"Label Number: {label_no-1}, Cohen Kappa Score: ", cohen_kappa_score(y_1, y_2))




