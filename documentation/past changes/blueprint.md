# Program Instructions: Spectroscopic Data Analysis and Exoplanet Characterization

## 1. Core Purpose
Develop a scientific application that collects, organizes, visualizes, and analyzes spectroscopic data from credible public sources and user-uploaded files. The program should support the comparison of molecular spectra with astronomical observations to investigate the possible chemical composition and physical properties of exoplanets and other astronomical objects.

## 2. Data Acquisition and Sources
- Retrieve publicly available scientific data from credible databases, APIs, and research institutions.
- Support importing user-uploaded data files, including relevant spectral data, light curves, and astronomical observation files.
- Integrate online molecular spectral databases to retrieve reference spectra for known molecules and compounds.
- Record source information, citations, observation metadata, measurement units, and relevant uncertainty information.
- Clearly distinguish measured observations, reference data, processed results, and theoretical predictions.

## 3. Data Storage and Reuse
- Store downloaded datasets locally in an organized, easily accessible directory structure.
- Maintain a searchable catalog of datasets, including their sources, retrieval dates, metadata, and processing history.
- Cache retrieved data and reuse existing local copies whenever possible instead of downloading the same datasets repeatedly.
- Allow users to update, replace, or refresh datasets when newer versions become available.
- Preserve original data and keep processed or transformed versions separate to ensure reproducibility.

## 4. Visualization and Metadata
- Provide an intuitive graphical interface for browsing targets, datasets, and analysis results.
- Generate interactive plots of spectral data, allowing users to compare multiple spectra on consistent axes.
- Display relevant metadata alongside graphs, including target identity, observation details, spectral range, resolution, units, source, and uncertainties when available.
- Allow users to select, overlay, normalize, align, and compare datasets without modifying the original measurements.
- Clearly label graphs, axes, data sources, and processing operations.

## 5. Molecular Spectral Analysis
- Retrieve reference spectra for molecules from established scientific databases.
- Allow users to select individual molecules or combinations of molecules for comparison with astronomical spectra.
- Support appropriate preprocessing methods, including wavelength or wavenumber alignment, unit conversion, normalization, resolution matching, and noise handling.
- Account for differences between laboratory reference spectra and astronomical observations, including instrumental resolution, temperature, pressure, and other relevant conditions when data are available.
- Identify spectral features that may correspond to known molecules while accounting for overlapping features, noise, and alternative explanations.

## 6. Exoplanet Spectrum Comparison
- Compare reference molecular spectra with observed or retrieved exoplanet spectra.
- Evaluate potential molecular signatures and investigate which compounds may explain observed spectral features.
- Where the available data and models support it, estimate or project possible atmospheric composition and physical properties.
- Distinguish qualitative feature matching from quantitative abundance estimates or atmospheric retrievals.
- Report uncertainties, assumptions, model limitations, and alternative interpretations. Do not present a possible molecular match as a confirmed detection.
- Support comparisons across multiple targets and observations where useful.

## 7. Improve Existing Tools
- Inspect and build upon the analysis tools already implemented in the program.
- Prioritize improving accuracy, consistency, reproducibility, and scientific validity rather than unnecessarily replacing working functionality.
- Standardize data formats, units, preprocessing methods, plotting conventions, and analysis outputs.
- Validate calculations and analytical methods against known results, reference data, or established scientific methods whenever possible.
- Add error handling, input validation, logging, and informative messages when data are missing, incompatible, or unreliable.
- Make analytical assumptions and processing steps transparent to the user.

## 8. Development Priorities
1. Inspect the existing codebase and identify current capabilities, limitations, and incomplete features.
2. Establish reliable data acquisition, metadata management, local storage, and caching.
3. Improve interactive visualization and dataset comparison.
4. Expand molecular spectral matching and exoplanet analysis.
5. Validate analytical methods and improve uncertainty reporting.
6. Maintain a modular, maintainable architecture that allows new data sources and analytical tools to be added later.

**General development principle:** Build a reliable, reproducible scientific analysis platform. Prioritize credible data, transparent methods, reusable datasets, and defensible conclusions. Avoid inventing data, overstating confidence, or adding unnecessary complexity. Preserve existing functionality unless there is a clear reason to change it.