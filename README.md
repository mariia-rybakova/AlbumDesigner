
# AI Automatic Album Design ADD
This repository represents the basline of the album design using AI algorithms For more detials see this [documentatuon](https://docs.google.com/document/d/1L4AZ2CfKlSDMcJzIG_QZs3Fje-yeK8j54V_i7g3Rb2A/edit#heading=h.vu35xyfup6qn)
to see some albums desgins here in [Figma](https://www.figma.com/design/GzUsjLWUur1Zmxl1AGLquI/Album-Design-Breakdown)

# Algorithms for ADD 
The AI album design consists of four important algorithms:

1. **Automatic Selection**
2. **Combining Data Structures**
3. **Creating the Album**
4. **Cropping**


# Automatic Selection:
This algorithm aims to select a number of images to be used in the album design process. The algorithm works as follows: given each image a score, this score is calculated by the product of several factors:

- People in the Image: The user selects the people they want to see in the image. The score is calculated based on the probability of the presence of these people in the image.

- Image Similarity: The user selects ten images. We compare the similarity between each of the ten images and each image in the gallery using the cosine similarity score.

- Image Class: Each image in the gallery has a specific class obtained from context clustering processing. We compare each image's class in the gallery with the ten images selected by the user by taking the probability of a match and adding a penalty if there is no match.

- Tag Cloud: The user selects a number of tags that represent their preferences. We use the similarity score between the image embeddings and the tags' features, which are computed and saved in a .pkl file.

- Ranking: Each image has a ranking score indicating its importance. The higher the score, the more important the image is.

The total score is the product of each of the scores explained above.

The selection process depends on the relationship of the user to the bride and groom, whether they are parents or friends. We have set several rules where each category of the wedding has a percentage related to the user's relationship. For example, parents might want to see more portraits of the whole family compared to friends. Based on the number of available images in each group, we consider the percentage needed for that category and select the highest-scoring images. The process is iterative over the clusters we have created until there are no further clusters to select from.

There are cases where we handle the event differently, such as in "settings," where identical images might be a bottleneck for selection. To avoid selecting two or three photos of the same setting view, we ignore images with the same cluster label and image class and select only one image from each similar cluster label.

#  Combining data structures:
In this part, where we collect the required data for the design process, we obtain image embeddings, context clustering, person detections, and image metadata such as central background, diameter, etc. We save the data in a DataFrame structure to make the process easier.

# Creating the album
Creating the album is based on grouping the data by two major factors: image time and image cluster class, which indicates the image content. This may result in a number of groups (N). We might encounter issues where the groups are too small (e.g., 2 or 3 images), making it difficult to fit them into one spread (2 pages), or too large, where we don't have a layout to handle such a group. In these cases, we address such problematic groups using the "Splitting & Merging" process. The large groups are split based on image embeddings and image times, while the small groups are combined with the closest group within the same main time group based on content similarity.

Once the groups are ready for album processing, the algorithm for this step is considered a tree graph. We need to select N spreads, then layouts, and finally the distribution of images within the layouts, until we reach the leaf of the tree where the calculations are made. Each possibility is given a score, and we select the highest score for each node, progressing up the tree until we finalize the decision on the number of spreads, the layouts, and the distribution of the images within the layouts.

The scores are based on:

- **Image Importance Ranking**: This score helps us determine which image will go into which box in the layout, as the ranking score indicates the image's importance. The size of the box also influences this, where we calculate the correlation between the importance ranking and box size, so a higher importance image will be placed in a bigger box.


- **Layout Orientation**: Layouts with complex orientations are less preferable compared to layouts with simple orientations. A layout with a unified orientation (e.g., landscape only or portrait only) is preferred. We penalize layouts with mixed orientations.


- **Black and White**: If a page layout contains a mix of color images, it will receive a penalty because our goal is to keep black-and-white images on separate pages from color images.


- **Category**: Each event has a mean and standard deviation, and we calculate a Gaussian score to control the number of images for each event.


- **Cropping**: Cropping is not applied 100% but is considered for squares. If cropping is needed, a penalty is added.


- **Layout Repetition Penalty**: If a layout is used more than 5 times, a penalty is applied to avoid repetition throughout the album.


# Cropping

The process of cropping involves removing the background from the foreground of an image and then detecting the people within the image. Here's how it works:

1. **Background Removal**: First, we remove the background to focus on the foreground elements of the image.

2. **People Detection**: Next, we detect the people within the image.

3. **Drawing a Circle**: We draw a circle around the detected people to ensure that all individuals are included within it.

4. **Cropping**: Anything outside this circle can be cropped based on the aspect ratio, provided that it does not intersect with the circle.

This approach ensures that all detected people remain in the cropped image while maintaining the aspect ratio and avoiding intersections with the circle.


# Deployment & Debugging:

1. **Install the repository on your local device**

2. **Create a virtual environment**

3. **Install the required packages**

4. **Run the `albumDesignService.py` file**


# Testing 

You can switch to branch called "optimized" where we do experiments and testing over our project methods. 








