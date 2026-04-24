import warnings

warnings.filterwarnings('ignore', category=UserWarning)
from utils import *
from PIL import Image, ImageDraw, ImageFont

scaling_factor = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_2.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_2_new.pth.tar"
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_2_single.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_2_single.pth.tar"
# srgan_checkpoint = "./checkpoint_srgan_8.pth.tar"
# srresnet_checkpoint = "./checkpoint_srresnet_8_new.pth.tar"
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_16.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_16_new.pth.tar"
srgan_checkpoint = "./checkpoints/checkpoint_srgan_4_new.pth.tar"
srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_4_new.pth.tar"
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_16.pth.tar"
# srresnet_checkpoint1 = "./checkpoints/checkpoint_srresnet_16_4.pth.tar"
# srresnet_checkpoint2 = "./checkpoints/checkpoint_srresnet_4_new.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))['model'].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator'].to(device)
srgan_generator.eval()


def visualize_sr(img):
    hr_img = Image.open(img, mode="r")  # (3, 128, 128)
    lr_img = hr_img.resize((int(hr_img.width / scaling_factor), int(hr_img.height / scaling_factor)), Image.BICUBIC)
    # (3, w/k, h/k)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Nearest Upsampling
    nearest_img = lr_img.resize((hr_img.width, hr_img.height), Image.NEAREST)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    # Create grid
    margin = 20
    grid_img = Image.new('RGB', (5 * hr_img.width + 9 * margin, hr_img.height + 3 * margin), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype(r"X:\Codes\SR-demo\times.ttf", size=25)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light
        # font installed if you have MS Office Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the "
            "function.")
        font = ImageFont.load_default()

    bbox = font.getbbox("map1")
    text_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    draw.text(xy=[0, 2 * margin + hr_img.height / 2 - text_size[1] / 2 - 5],
              text="map1",
              font=font,
              fill='black')

    # Place LR image
    # grid_img.paste(nearest_img, (4*margin, 0))
    grid_img.paste(nearest_img, (4 * margin, 2 * margin))
    bbox = font.getbbox("LR/nearest")
    text_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    draw.text(xy=[nearest_img.width / 2 - text_size[0] / 2 + 4 * margin, text_size[1] - 10],
              text="LR/nearest",
              font=font,
              fill='black')

    # # Place Bicubic image
    # grid_img.paste(bicubic_img, (int(nearest_img.width + 5 * margin), 0))
    grid_img.paste(bicubic_img, (int(nearest_img.width + 5 * margin), 2 * margin))
    bbox = font.getbbox("bicubic")
    text_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    draw.text(
        xy=[nearest_img.width + bicubic_img.width / 2 - text_size[0] / 2 + 5 * margin, text_size[1] - 10],
        text="bicubic",
        font=font,
        fill='black')

    # Place SRGAN image
    # grid_img.paste(sr_img_srgan, (int(nearest_img.width + bicubic_img.width + 6 * margin), 0))
    grid_img.paste(sr_img_srgan, (int(nearest_img.width + bicubic_img.width + 6 * margin), 2 * margin))
    bbox = font.getbbox("SRGAN")
    text_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    draw.text(
        xy=[nearest_img.width + bicubic_img.width + sr_img_srgan.width / 2 - text_size[0] / 2 + 6 * margin,
            text_size[1] - 10],
        text="SRGAN",
        font=font,
        fill='black')

    # Place SRResNet image
    # grid_img.paste(sr_img_srresnet,
    #                (int(nearest_img.width + bicubic_img.width + sr_img_srgan.width + 7 * margin), 0))
    grid_img.paste(sr_img_srresnet,
                   (int(nearest_img.width + bicubic_img.width + sr_img_srgan.width + 7 * margin), 2 * margin))
    bbox = font.getbbox("SRResNet")
    text_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    draw.text(
        xy=[nearest_img.width + bicubic_img.width + sr_img_srgan.width + sr_img_srresnet.width / 2 - text_size[
            0] / 2 + 7 * margin, text_size[1] - 10],
        text="SRResNet",
        font=font,
        fill='black')

    # Place HR image
    # grid_img.paste(hr_img,
    #                (int(nearest_img.width+bicubic_img.width+sr_img_srgan.width+sr_img_srresnet.width+8*margin), 0))
    grid_img.paste(hr_img,
                   (int(nearest_img.width + bicubic_img.width + sr_img_srgan.width + hr_img.width + 8 * margin),
                    2 * margin))
    bbox = font.getbbox("Ground Truth")
    text_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    draw.text(
        xy=[nearest_img.width + bicubic_img.width + sr_img_srgan.width + sr_img_srresnet.width + hr_img.width / 2 -
            text_size[
                0] / 2 + 8 * margin, text_size[1] - 10],
        text="Ground Truth",
        font=font,
        fill='black')

    # Display grid
    grid_img.show()
    grid_img.save('grid.png')


if __name__ == '__main__':
    visualize_sr('datasets/train_radio/0_19.png')
