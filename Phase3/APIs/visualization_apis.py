import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import plotly.offline
import plotly.graph_objs as go
import networkx as nx


def edge_generator(graph, n1_ind):
    row = graph[n1_ind]
    print("Node " + str(n1_ind))

    el = []
    for n2_ind, n2 in enumerate(row):
        if n2 == 1:
            el.append((str(n1_ind), str(n2_ind)))

    return el


def initializeNetwork(graph, num_nodes):
    '''
    Initializes the graph (nodes, edges) and creates edge trace/component for graphing.

    :param graph:
    :param num_nodes:
    :return:
    '''
    edge_list = []

    listOfEdgeLists = Parallel(n_jobs=multiprocessing.cpu_count() - 1, backend='loky')(
        delayed(edge_generator)(graph, i) for i in range(num_nodes))
    for l in listOfEdgeLists:
        edge_list.extend(l)

    DG = nx.DiGraph()
    DG.add_nodes_from(map(str, range(graph.shape[0])))
    DG.add_edges_from(edge_list)

    pos = nx.spring_layout(DG)

    # The edges will be drawn as lines:
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in DG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    return DG, pos, edge_trace


def visualize_groups(graph, labels, groupType):
    '''
    Create network graph given labels, where nodes colored by cluster/class

    :param graph:
    :param labels: np.array of ints representing either cluster # or class. NOTE: values should be from 0..max-cluster/class-number
    :param groupType: either 'cluster' or 'class' strings
    :return:
    '''
    num_nodes = graph.shape[0]
    imageIds = np.load('rows.npy')
    max_label = np.amax(labels)

    DG, pos, edge_trace = initializeNetwork(graph, num_nodes)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Rainbow',
            reversescale=True,
            color=labels.astype(np.float),
            size=10,
            colorbar=dict(
                thickness=20,
                # title='Labels',
                xanchor='left',
                titleside='bottom',
                showticklabels=True,
                tickmode="array",
                ticks='outside',
                tickvals=list(range(max_label + 1)),
                ticktext=[groupType + " #" + str(i) for i in range(max_label + 1)]
            ),
            line=dict(width=2)))

    for node in DG.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node in range(num_nodes):
        node_info = 'imageId: ' + imageIds[int(node)] + ' cluster: ' + str(labels[int(node)])
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title='Image Network',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plotly.offline.plot(fig)


def visualize_similar(graph, queryImageIds, simImageIds):
    '''
    Create network graph where K-most similar/dominant images are colored.
    Query nodes (which should by subset of similarImages) are same color as k-similar, BUT their node info
    which appears when hovering a node states "QUERY"

    :param graph:
    :param queryImageIds: list of imageIds of query nodes
    :param simImageIds: list of k-similar/dominant imageIds
    :return:
    '''
    num_nodes = graph.shape[0]
    imageIds = np.load('rows.npy')

    idToNode = {y: x for x, y in enumerate(imageIds)}
    queryImageIds = list(map(lambda x: idToNode[x], queryImageIds))
    simImageIds = list(map(lambda x: idToNode[x], simImageIds))

    DG, pos, edge_trace = initializeNetwork(graph, num_nodes)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=20,
                # title='Labels',
                xanchor='left',
                titleside='bottom',
                showticklabels=True,
                tickmode="array",
                ticks='outside',
                tickvals=[0, 1],
                ticktext=["other", "K-Most Dominant/Similar"]
            ),
            line=dict(width=2)))

    for node in DG.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node in range(num_nodes):
        node_info = 'imageId: ' + imageIds[int(node)]
        if node in queryImageIds:
            node_info += ' QUERY NODE'
            node_trace['marker']['color'] += tuple([1])
        elif node in simImageIds:
            node_trace['marker']['color'] += tuple([1])
        else:
            node_trace['marker']['color'] += tuple([0])

        node_trace['text'] += tuple([node_info])

    # Create figure:
    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title='Image Network',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plotly.offline.plot(fig)
