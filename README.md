
## References:

- check out vaex https://github.com/vaexio/vaex
- renaming submodules : https://stackoverflow.com/questions/4526910/rename-a-git-submodule
- semantic segmentation is the task we are trying to do
- `segmentation-model-pt` downloads models to ~/.cache/torch/hub in case you want to delete/look at it later
- https://github.com/Mstfakts/Building-Detection-MaskRCNN
- imp: image data in spacenet is pan-sharpened! it's a way of creating high-res images from multiple low-res satellite bands. Can look into this for later.
- have to disaple shapely cython speedups: https://github.com/Toblerity/Fiona/issues/383#issuecomment-589418248
- mention that peeps gotta clone with the submodules recursively
- updated masked-rcnn to tf2 via : https://github.com/matterport/Mask_RCNN/issues/1070#issuecomment-740430758

- torch mask-rcnn tutorial : https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#testing-forward-method-optional, and https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=c736ba0ec45ef51f2071aae01b05e46e4ec44338&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f416e647265772d4e672d732d6e756d6265722d6f6e652d66616e2f5079546f7263682d4b657261732f633733366261306563343565663531663230373161616530316230356534366534656334343333382f4e6f7465626f6f6b732f496d616765732f746f726368766973696f6e5f6f626a6563745f646574656374696f6e5f66696e6574756e696e672f696d6167655f746f726368766973696f6e5f6f626a6563745f646574656374696f6e5f66696e6574756e696e672e6970796e62&logged_in=false&nwo=Andrew-Ng-s-number-one-fan%2FPyTorch-Keras&path=Notebooks%2FImages%2Ftorchvision_object_detection_finetuning%2Fimage_torchvision_object_detection_finetuning.ipynb&platform=android&repository_id=290980616&repository_type=Repository&version=99
- can even use better(?) obj detection and segmentation models? masked-rcnn was from 2016 I think
- how img pan sharpening works : https://www.youtube.com/watch?v=-139c169pKQ

## Notes

PAN-sharpening: RGB-> HSV, swap the Value with panchromatic band (covers more wavelengths and thus has more details), add the original Hue and saturation, then convert back to RGB for a pan-sharpened image.
