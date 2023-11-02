![bvhsdk logo](logo.png)

- **Documentation:** https://bvhsdk.readthedocs.io/
- **Github:** https://github.com/rltonoli/bvhsdk

## Description:
bvhsdk is a small but versatile Python library designed for the management of BVH (Biovision Hierarchy) files commonly used in animation and motion capture. This library offers a comprehensive set of tools to read, write, and extract valuable information from BVH animations. Whether you're a computer graphics artist, animator, researcher, or developer, bvhsdk simplifies your BVH file-handling needs.

## Key Features:

- **BVH File I/O:** Easily read BVH files into Python data structures or write modified animations back to BVH format.

- **Animation Data Extraction:** Extract information from BVH files, such as joint hierarchies, frametime, and motion capture poses. Access motion data for further analysis or manipulation, such as local and global positions, rotation, and translation.

- **Manipulation Tools:** Modify BVH animations programmatically. Use bvhsdk to implement algorithms for retargeting animations, blending motions, or smoothing trajectories.

- Data Visualization (work in progress): Visualize BVH animations with a GUI tool for representations of motion data.

## Use Cases

- **Education:** Study, test and gain practical experience with several fundamental methods and concepts of 3D math for animation such as Euler angles, matrices and transformations, local and global coordinate systems, among others. 

- **Motion Capture Analysis:** Extract valuable insights from motion capture data for research, biomechanics, or character animation improvements.

- **Data Science:** Utilize motion data extracted from BVH files for machine learning, data analysis, or pattern recognition applications.

- **Game Development:** Incorporate BVH animations into video games for realistic character movements and interactive experiences.

## Featured in

- Tonoli, R. L., Marques, L. B. D. M. M., Ueda, L. H., & Costa, P. P. D. (2023, September). Gesture Generation with Diffusion Models Aided by Speech Activity Information. In GENEA: Generation and Evaluation of Non-verbal Behaviour for Embodied Agents Challenge 2023.

- Tonoli, R. L. (2019). Motion retargeting preserving spatial relationship: Transferência de movimentos preservando relação espacial (DISSERTAÇÃO Mestre em Engenharia Elétrica). Campinas, SP.

- Tonoli, R. L., Costa, P. D. P., & De Martino, J. M. Signing Avatar in Virtual Reality: An Intelligibility Study.

## Project Disclaimer

**Note:** This project is a work in progress and has been primarily developed by a single contributor. As a result, it may currently lack documentation, contain sections of code that are not fully optimized and include remnants from its origins as part of another project.

We appreciate your understanding and patience as we continue to improve and refine this project. We are actively working to address these issues and make the codebase more organized and accessible. If you have expertise in this area or are interested in contributing, your assistance would be greatly appreciated. You can check how you can contribute and some of the interesting improvements below. Together, we can make this project even better!

Thank you for your interest and support.

## Contribute

Want to contribute? Feel free to create pull requests, issues, or emails. Here are some **major** improvements that would greatly benefit bvhsdk:

- Support for all rotation orders; currently, only ZXY is fully supported.
- Documentation: help document every function and provide some examples when needed.
- Code: standardize code, such as variables and function names.
- Mathutils: include other representations of rotations and convertions between them, such as quaternions and axis-angle.
- GUI: improve current visualization; develop other visualizations, such as hierarchy, foot contact, velocity and acceleration plots; overall toolkits for visualization.
- Skeleton Map: include fingers; include other skeletons standards; create an option for reading and writing skeleton map.


## Contact

If you have any questions or suggestions, please contact us via: r105652@dac.unicamp.br

## Citation

If you find this library useful, please cite this master's thesis:

```
@mastersthesis{tonoli2019motion,
  title={Motion retargeting preserving spatial relationship: Transfer{\^e}ncia de movimentos preservando rela{\c{c}}{\~a}o espacial},
  author={Tonoli, Rodolfo Luis},
  year={2019},
  school={Universidade Estadual de Campinas, Faculdade de Engenharia Elétrica e de Computação},
  address={Campinas, SP}
  url={https://hdl.handle.net/20.500.12733/1639986}
}
```
