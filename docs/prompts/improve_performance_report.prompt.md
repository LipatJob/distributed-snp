I want you to improve on how we are collecting performance report data for our distributed SNP application. 

Here are some ideas to consider:
- getPerformanceReport should return a structured data format instead of plain text
- The metrics should be standardized across different implementations
- For CUDA implementations
  - Include kernel execution times
  - Memory transfer times
- For MPI implementations
  - Include communication times between nodes
  - Message sizes and counts

Feel free to suggest any other improvements that could enhance the performance reporting capabilities.

