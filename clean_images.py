from PIL import Image
import os, fnmatch

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    # pasting a downsized image inside a square image (final_size x final_size)
    # the location of pasting is a tuple (second argument) - shows a location of 
    # the upper left corner 
    new_im.paste(im, box=((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    path = "images/"
    dirs = os.listdir(path)
    final_size = 512
    directory_name = "cleaned_images"
    os.mkdir(directory_name)
    print("Directory " , directory_name ,  " Created ") 
    for n, item in enumerate(dirs):
        if n % 100 == 0:
            print(n, item)
        if 'csv' in item:
            continue
        im = Image.open('images/' + item)
        #  print(item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{directory_name}/{os.path.basename(item)}')
        # print(f'{directory_name}/{os.path.basename(item)}')

    
    # assert len(os.listdir("images/")) == len(os.listdir("cleaned_images/"))

#%%
# from PIL import Image
# im = Image.open('/Users/mariakosyuchenko/AI_Core/facebook-marketplaces-recommendation-ranking-system/images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg')
# # im.show()
# # %%
# size = im.size
# print(f"image size {size}")
# print(f"max size is {max(im.size)}")
# ratio = 512 / max(size)
# new_image_size = tuple([int(x*ratio) for x in size])
# print(f'new image size {new_image_size}')

# # %%
# final_size = 512
# im = im.resize(new_image_size, Image.ANTIALIAS)
# new_im = Image.new("RGB", (final_size, final_size))

# new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
# new_im.show()
# # %%
# (512-384)//2
# # %%
# path = "images/"
# dirs = os.listdir(path)

# # %%
# dirs
# %%
# len(os.listdir("cleaned_images/"))
path = "images/"
dirs = os.listdir(path)
print(dirs)

# %%
print(len(os.listdir("cleaned_images/")))
print(len(os.listdir("images/")))
# %%
