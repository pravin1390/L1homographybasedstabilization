This is an unofficial implementation of 

Bradley, Arwen, et al. "Cinematic-L1 video stabilization with a log-homography model." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.

Command to run the code:
python3 L1homography_lpp.py -i data/0.avi -o output/0_out.avi -crop_ratio 0.8

The code is still under construction:
1) In the outputs there is a slight distortion still visible
2) Keystone-translation ratio constraint is not added.
3) Running or fast moving videos are not properly stablilized. Need to check if the issue is with algorithm or implementation

Please feel free to contact if found any errors: pravinn@iisc.ac.in
