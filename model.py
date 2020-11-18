import torch.nn as nn
import torch
from utils import *
import matplotlib.pyplot as plt


# custom loss function based on MSELoss. Slightly biased towards more extreme (colorful) values
def MSEColorfulLoss(output, target):
    loss = torch.mean((output - target)**2 + 0.15 * torch.tanh(6*(torch.abs(target) - torch.abs(output) - 0.333)))
    return loss

class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        k = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=7//2),
            nn.ReLU(True),
            nn.BatchNorm2d(16)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=5//2),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=k, stride=2, padding=k//2),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=k, stride=1, padding=k//2),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )
        self.drop_out = nn.Dropout()
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=k, stride=2, padding=k//2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=k, stride=2, padding=k//2, output_padding=1),
            nn.ReLU(True)
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=k, stride=2, padding=k//2, output_padding=1),
            nn.ReLU(True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=k, stride=1, padding=k//2),
            nn.ReLU(True)
        )
        self.layer9 = nn.Conv2d(8, 2, kernel_size=5, stride=1, padding=5//2)
        
        
    # low level functionality
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.drop_out(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        return out
    
    def use(self, x):
        return self.forward((x-50)/100.0) * 110
    
    def test(self, x, y, criterion=MSEColorfulLoss):
        return criterion(self.forward((x-50)/100.0), y / 110.0).item()
        
    def train(self, train_loader, learning_rate=0.1, num_epochs=10, criterion=None, optimizer=None):
        
        if criterion == None:
            criterion = MSEColorfulLoss
            
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Train the model
        total_step = len(train_loader)
        loss_list = []
        for epoch in range(num_epochs):
            for i, (images, expected) in enumerate(train_loader):
                # Run the forward pass
                outputs = self.forward((images-50)/100.0)
                loss = criterion(outputs, expected / 110.0)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ((epoch + 1) % 10 == 0 or epoch < 5) and i == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, loss.item()))
                    
    def save(self, filename="model_weights.pt"):
        torch.save(self.state_dict(), filename)
        
    def load(self, filename="model_weights.pt"):
        self.load_state_dict(torch.load(filename))
        #self.eval()
        
        
    # high level functionality
    
    # trains from images within a directory (not subdirectories), and optionally returns loss from a test set
    def train_from_dir(self, train_dir, test_dir=None, learning_rate=0.001, num_epochs=200, batch_size=10, shuffle_data=True, criterion=None, model_filename=None):
        train_data = load_images(get_files_in_dir(train_dir))

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle_data)
        self.train(train_loader, learning_rate=learning_rate, num_epochs=num_epochs, criterion=criterion)
        
        if (model_filename != None):
            self.save(model_filename)
        
        
        if (test_dir != None):
            test_x, test_y = separate_io(load_images(get_files_in_dir(test_dir)))
            return self.test(test_x, test_y)
        
        return None
    
    def colorize(self, img_path, out_path=None, show_figure=True):
        img = load_img(img_path)
        (tens_l_orig, tens_l_rs, tens_ab_rs) = preprocess_img(img, HW=(256,256))

        model_output = self.use(tens_l_rs).cpu()

        out_img_color = postprocess_tens(tens_l_orig, model_output)
        out_img_orig  = postprocess_tens(tens_l_orig, tens_ab_rs)

        img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

        if (out_path != None):
            plt.imsave(out_path, out_img_color)

        if (show_figure):
            plt.figure(figsize=(15,8))
            plt.subplot(2,3,1)
            plt.imshow(img)
            plt.title('Original')
            plt.axis('off')

            plt.subplot(2,3,2)
            plt.imshow(img_bw)
            plt.title('Input')
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.imshow(out_img_color)
            plt.title('Output')
            plt.axis('off')
            plt.show()