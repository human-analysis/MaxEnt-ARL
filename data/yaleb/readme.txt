Each set_#.pdata contains the training images and the person ID from that specific lightning orientation (its a pickle file). 
The only preprocessing that was done was normalization of the pixels to [0, 1]. 

The test.pdata contains the test data which are organized as follows: 'x' are the images, 't' contains the person ID and 'light' denotes the lightning orientation. 
We only kept the test images where light != 5.
