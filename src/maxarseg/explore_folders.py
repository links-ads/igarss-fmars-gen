import os

def create_pre_post_diz_set(event_id, root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/'):
    """
    Create a dictionary of sets for pre and post folders.
    The key is the mosaic name and the value is a set containing
    the images in that mosaic 
    """
    pre_path = os.path.join(root, event_id, 'pre')
    #print(pre_path)
    post_path = os.path.join(root, event_id, 'post')
    #print(post_path)
    pre_diz = {}
    try:
        for pre_mosaic in os.listdir(pre_path):
            #print('pre_mosaic', pre_mosaic)
            pre_diz[pre_mosaic] = set()
            for pre_img in os.listdir(os.path.join(pre_path, pre_mosaic)):
                pre_diz[pre_mosaic].add(pre_img)
    except:
        print('No pre folder')
    
    post_diz = {}
    try:
        for post_mosaic in os.listdir(post_path):
            #print('post_mosaic', post_mosaic)
            post_diz[post_mosaic] = set()
            for post_img in os.listdir(os.path.join(post_path, post_mosaic)):
                post_diz[post_mosaic].add(post_img)
    except:
        print('No post folder')
    
    return pre_diz, post_diz

def subtraction_between_diz(matching, pre_diz, post_diz):
    """
    Function not to be used outside the check_matching_pre_post() function.
    """

    non_matching = {}

    for k_pre in pre_diz.keys():
        non_matching[k_pre] = pre_diz[k_pre]
        if k_pre in matching.keys():
            non_matching[k_pre] -= matching[k_pre]
        
    for k_post in post_diz.keys():
        non_matching[k_post] = post_diz[k_post]
        if k_post in matching.keys():
            non_matching[k_post] -= matching[k_post]
    tmp = {}        
    for k in non_matching.keys():
        if len(non_matching[k]) != 0:
            tmp[k] = non_matching[k]
    non_matching = tmp
    return non_matching

def check_matching_pre_post(event_id, root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/', verbose=True):
    """
    Params:
    event_id example: 'Gambia-flooding-8-11-2022'
    
    Returns:
    - matching: a dictionary with the mosaic name as key and the
        set of matching images contained in that mosaic as value
    - non_matching: a dictionary with the mosaic name as key and the
        set of non matching images in that mosaic as value
        (useful if you want to delete non matching images)
    - diz_img_mosaic: a dictionary with the image name as key and the
        set of mosaics that contain that image as value
    """
    
    pre_diz, post_diz = create_pre_post_diz_set(event_id, root = root)

    if verbose:
        print('Pre')
        for k in pre_diz.keys():
            print('-',k, '#img:',len(pre_diz[k]))
        print('\nPost')
        for k in post_diz.keys():
            print('-',k, '#img:',len(post_diz[k]))
    
    matching = {} #un diz con chiave il nome del mosaico e valore una lista contenente le immagini che matchano
    diz_img_mosaic = {} #un diz con chiave il nome dell'immagine e valore il nome dei mosaici a cui appartiene

    for k_pre in pre_diz.keys(): #Per ogni mosaico pre
        for k_post in post_diz.keys(): #controlla ogni mosaico post
            for img_post in post_diz[k_post]: # in particolare controlla ogni immagine post
                if img_post in pre_diz[k_pre]: #se l'immagine post è presente nel mosaico pre
                    #print(f'{img_post} è presente nel pre e nel post')
                    if k_pre not in matching.keys():
                        matching[k_pre] = set()
                    if k_post not in matching.keys():
                        matching[k_post] = set()
                    matching[k_pre].add(img_post)
                    matching[k_post].add(img_post)
                    if img_post not in diz_img_mosaic.keys(): 
                        diz_img_mosaic[img_post] = set()
                    diz_img_mosaic[img_post].add(k_pre)
                    diz_img_mosaic[img_post].add(k_post)

    non_matching = subtraction_between_diz(matching, pre_diz, post_diz)
    return matching, non_matching, diz_img_mosaic

def count_tif_files(path):
    """
    Given a path, this function explores all the subfolders and counts the number of .tif files.
    """
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tif"):
                count += 1
    return count

def compute_stats_on_event(event_id, root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/'):
    """
    Params:
    event_id: Example: "Gambia-flooding-8-11-2022"
    
    Returns:
    No returns but it prints:
    - print the number of images in the pre and post folders
    - count the number of matching and non matching images
    """

    matching, non_matching, diz_img_mosaic = check_matching_pre_post(event_id, root = root)

    matching_count = 0
    for k in matching.keys():
        matching_count += len(matching[k])
    total_tif = count_tif_files(os.path.join(root, event_id))

    print(f'\nMatching: {matching_count} images. {100 * matching_count/total_tif:.2f}%')

    non_matching_count = 0
    for k in non_matching.keys():
        non_matching_count += len(non_matching[k])

    print(f'\nNon matching: {non_matching_count} images. {100 * non_matching_count/total_tif:.2f}%')
