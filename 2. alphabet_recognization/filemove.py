import os, shutil

original_dataset_dir = './alphabet_images/'
base_dir = './alphabet_small'

train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#letter = 'A'
for letter in letters:
    original_A = os.path.join(original_dataset_dir,letter)
    
    train_A_dir = os.path.join(train_dir, letter)
    validation_A_dir = os.path.join(validation_dir, letter)
    test_A_dir = os.path.join(test_dir, letter)
    os.mkdir(train_A_dir)
    os.mkdir(validation_A_dir)
    os.mkdir(test_A_dir)
    
    fnames = [letter+'-{}.png'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_A, fname)
        dst = os.path.join(train_A_dir, fname)
        shutil.copyfile(src,dst)
        
    fnames = [letter+'-{}.png'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(original_A, fname)
        dst = os.path.join(validation_A_dir, fname)
        shutil.copyfile(src,dst)
        
    fnames = [letter+'-{}.png'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(original_A, fname)
        dst = os.path.join(test_A_dir, fname)
        shutil.copyfile(src,dst)