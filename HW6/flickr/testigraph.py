#!/usr/bin/python -w
import csv, igraph, cairo

reader = csv.reader(open("data/Flickr_sampled_edges/edges_sampled_map_35K.csv"))
G = igraph.Graph.TupleList(reader)

print G.ecount()

igraph.plot(G)

vdendrogram=G.community_edge_betweenness( clusters=150, directed=False, weights=None)
clusters = vdendrogram.as_clustering()
membership = clusters.membership

writer = csv.writer(open("data/Flickr_sampled_edges/community_membership.csv", "wb"))
for i,v in enumerate(membership):
    writer.writerow([i, v])
