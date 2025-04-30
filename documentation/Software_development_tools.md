## Software de development of tools
**Project:** *Catalan Music Classification and Analysis*   
**Contributors:** Adrià Cortés Cugat, Martí Armengol Ayala, Sergi De Vicente, Guillem Garcia, Jofre Geli, Javier Echávarri, Alex Salas Molina  
**Course:** Music Technology  

---

### Code Repository  

Link to our repository:
https://github.com/martiarmengol/MTG-102 

The project is managed collaboratively through a public GitHub repository, ensuring transparent access for all team members and instructors. The repository is structured to support modular development, version control, and reproducibility of all experiments and results.

- **code/**: Contains all Python scripts and Jupyter notebooks related to audio processing, embedding extraction (using CLAP), clustering algorithms (K-Means), classification (KNN), and data visualization (PCA, UMAP, t-SNE). It is organized to allow independent testing and reuse of each component.
- **database_csv/**: Stores .csv files with metadata and extracted audio features (BPM, instrumentation, genre, voice gender, acoustic v.s. Electronic, and YT Link). These structured datasets are used as inputs for machine learning models and for clustering/visualization tasks.
- **db_downloaded_songs/**: Contains the audio files used in the project. This folder includes organized .mp3 files grouped by time period. If hosting limitations apply, the repository links to external storage (Google Drive) while maintaining file structure and naming consistency.
- **documentation/**: Includes all written reports and supporting documents, such as the State of the Art, Software Development Tools, and others. It serves as a central location for academic and technical documentation
- **.gitattributes and .gitignore**: These configuration files manage cross-platform consistency (line endings) and ensure sensitive or unnecessary files (cache, large binaries) are excluded from version control. This supports clean collaboration and reproducible environments.
  
