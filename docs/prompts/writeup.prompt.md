# Role
Act as an academic practitioner with specialization on Membrane Computing and High-Performance Computing.

# Task
Create a technical paper describing the design, implementation, and performance analysis of the "Distributed SNP" project found in the current workspace. The paper should focus on the high-performance simulation framework that leverages hybrid MPI+CUDA architecture for simulating large-scale Spiking Neural P Systems.

# Context & Constraints
1.  **Audience:** The write-up should be tailored for graduate students and researchers in computer science and computational neuroscience who have a foundational understanding of neural networks but may not be familiar with SNP Systems.
2.  **Structure:** The write-up should include the following sections:
    * **Introduction:** Introduce Spiking Neural P Systems and the computational challenges of simulating them at scale.
    * **System Architecture:** Describe the modular design of the codebase, specifically the separation between the Linear Algebra engine and the SNP Simulator logic.
    * **Implementation Details:**
        *   **Matrix Representation:** Explain the mathematical model used ($C(k+1) = C(k) + Sp(k) \times M$) and its mapping to GPU kernels.
        *   **Hybrid Backend:** Detail the MPI+CUDA approach, including the Structure of Arrays (SoA) data layout on GPU and global state management via MPI.
        *   **Graph Partitioning:** Explain the implemented strategies: Linear, Louvain, and Red-Blue Pebbling, and their role in minimizing inter-node communication.
    *   **Case Study: Natural Number Sorting:** Describe the specific SNP topology (Input, Sorter, Output layers) and rule set used for sorting integers as implemented in `src/sort/SnpSort.cpp`.
    *   **Performance Evaluation:** Discuss the benchmark results (from `benchmark_results/`), analyzing the scalability, the $O(M \cdot N^2)$ complexity, and the impact of communication overhead.
    *   **Conclusion:** Summarize the key contributions of this distributed framework and suggest future optimizations.
3.  **References:** Include citations from reputable sources such as academic journals, conference papers, and authoritative books. Use a consistent citation style (e.g., APA, IEEE). Some examples include:
    * Martínez-del-Amor, M.Á.; Orellana-Martín, D.; Pérez-Hurtado, I.; Cabarle, F.G.C.; Adorna, H.N. Simulation of Spiking Neural P Systems with Sparse Matrix-Vector Operations. Processes 2021, 9, 690. https://doi.org/10.3390/pr9040690 (2006). Spiking Neural P Systems. Fundamenta Informaticae, 71(2-3), 279-308.
    * Ionescu, M.; Sburlan, D. Some Applications of Spiking Neural P Systems. Comput. Inform. 2008, 27, 515–528
    * Martínez-del-Amor, M.Á.; Orellana-Martín, D.; Cabarle, F.G.C.; Pérez-Jiménez, M.J.; Adorna, H.N. Sparse-matrix representation of spiking neural P systems for GPUs. In Proceedings of the 15th Brainstorming Week on Membrane Computing, Sevilla, Spain, 31 January–5 February 2017; pp. 161–170.
    * Zeng, X.; Adorna, H.; Martínez-del-Amor, M.A.; Pan, L.; Pérez-Jiménez, M.J. Matrix Representation of Spiking Neural P Systems. In Proceedings of the 11th International Conference on Membrane Computing, Jena, Germany, 24–27 August 2010; Volume 6501, pp. 377–391
5.  **Clarity and Precision:** Ensure that the language is clear, precise, and free of jargon. Use diagrams or illustrations where necessary to enhance understanding.
6.  **Originality:** The content must be original and not plagiarized. Ensure proper paraphrasing and citation of sources.
7.  **Formatting:** Use appropriate headings, subheadings, bullet points, and numbering to organize the content effectively.
