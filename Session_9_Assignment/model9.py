class Ultimus(nn.Module):
    def __init__(self):
        super(Ultimus, self).__init__()
        self.fc_for_query = nn.Linear(in_features=48, out_features = 8, bias = False)

        self.fc_for_key = nn.Linear(in_features = 48 ,out_features = 8, bias = False)

        self.fc_for_value = nn.Linear(in_features = 48, out_features = 8, bias = False)

        self.fc_out = nn.Linear(in_features = 8, out_features = 48, bias = False)

    def forward(self, x):

        query = self.fc_for_query(x)
        key = self.fc_for_key(x)
        value = self.fc_for_value(x)

        # print(query.T.shape, key.shape, value.shape)
        am = torch.matmul(query.T , key)
        am = F.log_softmax(am, dim = 1)
        am = am / 2.8284 ##### dividing by root(8)

        Z = torch.matmul(value, am)
        Z = self.fc_out(Z)

        return Z
        # print(Z.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(32)
        )
        
        self.ultimus1 = Ultimus()
        self.ultimus2 = Ultimus()
        self.ultimus3 = Ultimus()
        self.ultimus4 = Ultimus()

        self.fc_out = nn.Linear(in_features = 48, out_features = 10, bias = False)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,48)

        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)

        x = self.fc_out(x)

        return x