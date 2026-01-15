/*
 * Dijkstra's Shortest Path Algorithm (N=1000 vertices)
 * Tests: Control flow, branch prediction, irregular memory access
 * Expected runtime: ~4 hours on O3CPU (complex workload)
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define V 1000  // Number of vertices

// Static allocation for large arrays
static int graph[V][V];
static int dist[V];
static bool sptSet[V];

// Find vertex with minimum distance
int minDistance(void) {
    int min = INT_MAX;
    int min_index = 0;
    int v;
    
    for (v = 0; v < V; v++) {
        if (sptSet[v] == false && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    
    return min_index;
}

// Dijkstra's algorithm
void dijkstra(int src) {
    int count, u, v;
    
    // Initialize distances and sptSet
    for (v = 0; v < V; v++) {
        dist[v] = INT_MAX;
        sptSet[v] = false;
    }
    
    dist[src] = 0;
    
    // Find shortest path for all vertices
    for (count = 0; count < V - 1; count++) {
        u = minDistance();
        sptSet[u] = true;
        
        // Update dist values of adjacent vertices
        for (v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && 
                dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}

int main(void) {
    int i, j;
    int checksum = 0;
    
    // Initialize graph (sparse graph with some edges)
    for (i = 0; i < V; i++) {
        for (j = 0; j < V; j++) {
            graph[i][j] = 0;
        }
    }
    
    // Create edges: linear chain + shortcuts every 10 vertices
    for (i = 0; i < V - 1; i++) {
        graph[i][i + 1] = 1;  // Edge to next vertex
        graph[i + 1][i] = 1;  // Bidirectional
        
        // Add shortcuts
        if (i % 10 == 0 && i + 10 < V) {
            graph[i][i + 10] = 5;
            graph[i + 10][i] = 5;
        }
    }
    
    // Run Dijkstra from vertex 0
    dijkstra(0);
    
    // Compute checksum
    for (i = 0; i < V; i++) {
        if (dist[i] != INT_MAX) {
            checksum += dist[i];
        }
    }
    
    printf("Dijkstra's algorithm complete\n");
    printf("Checksum: %d\n", checksum);
    printf("Distance to vertex %d: %d\n", V-1, dist[V-1]);
    printf("Distance to vertex %d: %d\n", V/2, dist[V/2]);
    
    return 0;
}