import torch as torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sksurv.util import Surv
import warnings
warnings.filterwarnings('ignore')


class SurvivalLVQ(torch.nn.Module, BaseEstimator):
    def __init__(self, n_prototypes=2, n_omega_rows=None, batch_size=128,
                 init='kmeans', device=torch.device("cpu"), lr=1e-3, epochs=50, verbose=True, random_state=None):
        """
        Initializes the SurvivalLVQ model.

        Args:
            n_prototypes (int): The number of prototypes to learn.
            n_omega_rows (int, optional): The number of rows for the omega matrix, which determines the
                                          rank of the relevance matrix. If None, it's set to the number
                                          of features (full-rank). A smaller value can act as regularization.
            batch_size (int): The size of mini-batches for training.
            init (str): The initialization method for prototypes. 'kmeans' uses K-Means cluster centers,
                        'random' initializes them randomly.
            device (torch.device): The device to run the model on ('cpu' or 'cuda').
            lr (float): The learning rate for the optimizer.
            epochs (int): The number of training epochs.
            verbose (bool): If True, prints the loss at each epoch.
            random_state (int, optional): Random seed for reproducibility. If None, uses random initialization.
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.n_omega_rows = n_omega_rows        # dim of omega
        self.init = init
        self.verbose = verbose
        self.batch_size = batch_size
        self.random_state = random_state

    # initializes the model
    def _init_model(self, datapoints, D, T):
        """
        Initializes all model components, parameters, and pre-computes necessary values.
        This is called once at the beginning of the `fit` method.

        Args:
            datapoints (torch.Tensor): The input feature data.
            D (torch.Tensor): The event/censoring indicator (1 for event, 0 for censored).
            T (torch.Tensor): The time to event or censoring.
        """
        self.n_features = datapoints.shape[1]   # dim of features
        self.datapoints = datapoints
        self.D = D                              # censoring indicator
        self.T = T                              # time

        # sets the dim of omega to be the max(row, col) of datapoints
        if self.n_omega_rows is None:
            if datapoints.size(0) > datapoints.size(1):
                self.n_omega_rows = datapoints.size(1)
            else:
                self.n_omega_rows = datapoints.size(0)
        else:
            self.n_omega_rows = self.n_omega_rows

        # use only non-censored and unique event times
        self.timepoints = torch.unique(T[D > 0], sorted=True) 
        # subsample times if we have more than a 1000.
        if self.timepoints.size(0) > 1000:
            self.timepoints = self.timepoints[1::np.floor(self.timepoints.size(0) / 1000).astype('int')]
            # np.floor(2.8) = 2 
            # [1:: ...] represents the step size for slicing. for example, [1::2] = [1, 3, 5, 7, ...]
            # [starts: stops: step]
        self.n_timepoints = self.timepoints.size(0) # number of timepoints
        self.eig_vec = None 

        # impute missing values with median, for k-means init.
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        datapoints_imputed = imp.fit_transform(self.datapoints.numpy())
        Y = Surv().from_arrays(self.D.numpy().astype('?'), self.T.numpy())

        if self.n_omega_rows < self.n_features:
            # If n < p, use the pseudo inverse to approximate the identity matrix for lambda:
            # (omega^T)^-1 * omega^T * omega =  (omega^T)^-1 * I -->
            # omega = (omega^T)^-1 = approx I
            if self.random_state is not None:
                generator = torch.Generator(device=self.device).manual_seed(self.random_state)
                self.omega = torch.rand(size=(self.n_omega_rows, self.n_features), dtype=torch.float32,
                                        device=self.device, generator=generator)
            else:
                self.omega = torch.rand(size=(self.n_omega_rows, self.n_features), dtype=torch.float32,
                                        device=self.device)
            self.omega = torch.pinverse(self.omega.T)
        else:
            # use identity matrix as initial omega and thus lambda: every feature is weighted equally at start.
            self.omega = torch.eye(self.n_features, dtype=torch.float32, device=self.device)

        if self.init == 'kmeans':
            # fit kmeans on datapoints
            pass
            clf = KMeans(n_clusters=self.n_prototypes, n_init=1, random_state=self.random_state).fit(datapoints_imputed)

            # set initial prototype locations to the k-means.
            self.w = torch.tensor((clf.cluster_centers_), dtype=torch.float32, device=self.device)

        else:
            # random init of prototypes
            if self.random_state is not None:
                generator = torch.Generator(device=self.device).manual_seed(self.random_state)
                self.w = torch.mean(torch.tensor(datapoints_imputed, dtype=torch.float32), axis=0) * 0.01 * \
                         torch.randn(self.n_prototypes, datapoints.size(dim=1), dtype=torch.float32, device=self.device, generator=generator)
            else:
                self.w = torch.mean(torch.tensor(datapoints_imputed, dtype=torch.float32), axis=0) * 0.01 * \
                         torch.randn(self.n_prototypes, datapoints.size(dim=1), dtype=torch.float32, device=self.device)

        # some pytorch admin.
        self.datapoints = self.datapoints.to(self.device)
        self.omega = torch.nn.Parameter(self.omega)
        self.w = torch.nn.Parameter(self.w)


        # calculate the estimator of the conditional survival function of the censoring times
        # using the Kaplan-Meier method. Required for IPCW weighting
        x, y = self.estimate_kaplan_meier((self.D - 1) * -1, T)
        x = x.numpy()
        y = y.numpy()
        f = interp1d(x, y, bounds_error=True)

        self.G_t = f(T.numpy())
        self.G_t_unique = f(self.timepoints.numpy())
        self.G_t = torch.tensor(self.G_t, dtype=torch.float32)
        self.G_t_unique = torch.tensor(self.G_t_unique, dtype=torch.float32)

        self.IPCW_weights = self._calc_IPCW_weights(self.timepoints, self.D, self.T, self.G_t, self.G_t_unique)
        self.individual_curves = T[:, None] > self.timepoints

        self.c = self._calc_labels()
        self.normalize_trace() # make the total feature relevance sum = 1.

    def estimate_kaplan_meier(self, d, t, weights=None):
        """
        Calculates the Kaplan-Meier survival curve estimator.
        Can handle weighted samples.

        Args:
            d (torch.Tensor): Event indicator (1 for event, 0 for censored).
            t (torch.Tensor): Time to event/censoring.
            weights (torch.Tensor, optional): Sample weights. Defaults to uniform weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of unique times and corresponding survival probabilities.
        """
        if weights is None:
            weights = torch.ones_like(d)

        t_sorted, indices = torch.sort(t)
        d_sorted = d[indices]
        weights_sorted = weights[indices]
        t_unique, t_counts = torch.unique(t, return_counts=True)
        t_group = torch.repeat_interleave(t_counts)
        d_i = torch.bincount(t_group, weights=weights_sorted * d_sorted)  # number of events at time i
        n_i = torch.cumsum(torch.bincount(t_group, weights=weights_sorted).flip(-1), -1).flip(-1)
        surv = (1 - d_i / n_i).cumprod(0)
        return t_unique, surv

    def lambda_mat(self):
        """
        Computes the relevance matrix Lambda = Omega^T * Omega.
        """
        return self.omega.T @ self.omega

    def fit_labels(self):
        """
        Recalculates the prototype labels (survival curves). Called at the start of each epoch.
        """
        self.c = self._calc_labels()

    # fuction to calculate the IPCW weights for each timepoint in timepoints and sample in D and T
    # note that last weights are ill defined if last (in time) datapoint is censored
    def _calc_IPCW_weights(self, timepoints, D, T, G_t, G_t_unique):
        T_i_smaller_eq_t = T[:, None] <= timepoints
        T_i_greater_t = ~T_i_smaller_eq_t

        greater_equal_t = T_i_smaller_eq_t * (D / G_t)[:, None]
        greater_equal_t = torch.nan_to_num(greater_equal_t)

        smaller_t = T_i_greater_t * (1.0 / G_t_unique)[None, :]
        smaller_t = torch.nan_to_num(smaller_t)

        alpha = greater_equal_t + smaller_t
        return alpha

    # function to set the prototype labels
    def _calc_labels(self):
        # prepare the data for the calculation of the labels
        x = self.datapoints                # data points
        w_local = self.w.detach()          # prototypes, detach to not mess with the gradient graph(back-prop)
        omega_local = self.omega.detach()  # relevance matrix, detach to not mess with the gradient graph(back-prop)
        
        # compute distance between data points and prototypes
        d = self._omega_distance(x, w_local, omega_local) 
        
        # convert the distance to assignment probability
        q_ij = self._q_ij(d).cpu()

        # find the prototype with the highest assignment probability for each data point
        max_prob = q_ij.max(dim=0).values         # max probability for each data point
        mask = (q_ij == max_prob.unsqueeze(0))    # True if prototype wins
        q_ij = mask * 1.0                         # winner takes all

        # weighted Kaplan-Meier per prototype
        kap_mei_w = torch.zeros((self.n_prototypes, self.n_timepoints)) # initialize the survival curve for each prototype
        proto_assign = q_ij > 0  # convenience boolean mask 
        for i in range(w_local.size(0)):
            # check if at least two points are assigned. otherwise ignore prototype
            if torch.unique(self.T[proto_assign[i, :]]).size(0) < 2:  
                continue # skip inactive prototype(fewer than 2 events assigned)
            
            # compute the survival curve for the prototype
            x, y = self.estimate_kaplan_meier(
                self.D[proto_assign[i, :]],             # censoring indicator
                self.T[proto_assign[i, :]],             # observed time
                weights=q_ij[i, proto_assign[i, :]]     # = 1 for assigned data points
            )
            # interpolate the survival curve to a common grid
            f = interp1d(x.numpy(), y.numpy(), fill_value=(1.0, 0), bounds_error=False)  # y.numpy().min()
            # fill_value=(1.0, 0) means that if the time is greater than the maximum time, 
            # the survival probability is 1.0, and if the time is less than the minimum time, the survival probability is 0.0.
            kap_mei_w[i] = torch.tensor(f(self.timepoints), dtype=torch.float32)

        return kap_mei_w

    # predict the individual survival curve functions based on closest two prototypes.
    def predict_survival_function(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            pred = self.predict_curves(x)
            return [interp1d(self.timepoints.numpy(), pred[i], bounds_error=False, fill_value=(0.0, 1.0)) for i in
                    range(x.size(dim=0))]

    # 'risk score' for scikit learn
    def predict(self, X, closest=False):
        if closest:
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32)
                X = X.to(self.device)  # Fix: assign the result back to X
                w = self.w
                omega = self.omega
                d = self._omega_distance(X, w, omega)
                return d.cpu().min(dim=0).indices.numpy()
        else:
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32)
                pred = self.forward(X)
                return pred.numpy()

    # 'risk score' (inverse area under the survival curve).
    def forward(self, x):
        pred = self.predict_curves(x)
        res = -torch.trapezoid(pred, self.timepoints) / self.timepoints.max()
        if torch.sum(torch.isnan(res) > 0):
            pass
        return res

    def predict_closest(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)  # Fix: assign the result back to X
            w = self.w
            omega = self.omega
            d = self._omega_distance(X, w, omega)
            return d.cpu().min(dim=0).indices.numpy()

    # Finding closest prototypes
    def _q_ij(self, d):
        """
        Calculates the assignment probability for a sample to its two closest prototypes.
        p_j = d_k / (d_j + d_k), where j and k are the two closest prototypes.
        This means the closer a point is to a prototype, the higher its assignment probability.
        
        Args:
            d (torch.Tensor): Distances from samples to prototypes, shape (n_prototypes, n_samples).

        Returns:
            torch.Tensor: Assignment probabilities, shape (n_prototypes, n_samples).
        """
        top2_val, top2_i = d.topk(2, axis=0, largest=False)
        # flip the distances with flipud to make pj = dk / (dj + dk) and vice versa.
        p = torch.flipud(top2_val) / top2_val.sum(dim=0)[None, :]
        res = torch.scatter(torch.zeros_like(d), dim=0, index=top2_i, src=p)
        return res

    # the learnable distance function
    def _omega_distance(self, x1, w, omega):
        """
        Calculates the learnable distance d^2 = (x - w) * Lambda * (x - w)^T
        where Lambda = omega^T * omega.

        Args:
            x1 (torch.Tensor): Data points.
            w (torch.Tensor): Prototypes.
            omega (torch.Tensor): Relevance matrix component.

        Returns:
            torch.Tensor: The squared distances.
        """
        diff = x1 - w[:, None, :] 
        # w[:, None, :] is a 2D tensor with shape (n_prototypes, 1, n_features)

        # set nan diffs to 0 for NaN-LVQ
        diff = torch.nan_to_num(diff)
        dists = (diff @ omega.transpose(dim0=-2, dim1=-1) @ omega)[:, :, None, :] @ diff[:, :, :, None]
        # [:, :, None, :] is a 3D tensor with shape (n_prototypes, n_samples, 1, n_features)
        # [:, :, :, None] is a 3D tensor with shape (n_prototypes, n_samples, n_features, 1)
        return dists.squeeze(-1).squeeze(-1) # .squeeze(-1).squeeze(-1) is to remove the last two dimensions of the tensor

    def normalize_trace(self):
        """Normalize the matrix such that the trace is 1.
        shape of `mat`: (input_dim, mapping_dim) or (k, input_dim, mapping_dim)
        """
        with torch.no_grad():
            omega = self.omega.to(self.device)
            trace_norm = omega / torch.sqrt(torch.trace(omega.t() @ omega))
            self.omega.copy_(trace_norm.cpu())

    def predict_curves(self, x):
        x = x.to(self.device)
        w = self.w
        omega = self.omega
        d = self._omega_distance(x, w, omega)
        q_ij = self._q_ij(d)
        c = self.c.to(self.device)
        pred = q_ij[:, :, None] * c[:, None, :]
        pred = pred.sum(dim=0)
        return pred.cpu()

    def loss_brier(self, x, t=None, d=None):
        """
        Calculates the Brier score loss, a measure of the accuracy of probabilistic predictions.
        It is defined as the mean squared error between the predicted survival probabilities and the actual survival status.
        """
        # batch mode
        if t is None and d is None:
            x = self.datapoints
            IPCW_weights = self.IPCW_weights / x.size(0)
            individual_curves = self.individual_curves

        # mini-batch mode
        else:
            x_kap, y_kap = self.estimate_kaplan_meier((d - 1) * -1, t)
            x_kap_num = x_kap.numpy()
            y_kap_num = y_kap.numpy()
            f = interp1d(x_kap_num, y_kap_num, bounds_error=True)
            G_t = f(t.numpy())

            G_t = torch.tensor(G_t, dtype=torch.float32)
            G_t_unique = self.G_t_unique

            IPCW_weights = self._calc_IPCW_weights(self.timepoints, d, t, G_t, G_t_unique) / x.size(0)
            individual_curves = t[:, None] > self.timepoints # True if time is greater than timepoint   

        # calculate the predicted survival probabilities
        pred = self.predict_curves(x)
        # calculate the Brier score loss
        mu = IPCW_weights * (pred - (individual_curves * 1.0)) ** 2
        mu = mu.sum(dim=1)
        # return the mean of the Brier score loss
        return mu.mean()

    # fit method
    # X = data
    # y[0] = delta
    # y[1] = time
    def fit(self, X, y):
        """
        Fits the model to the training data.
        Updating the prototypes and the relevance matrix.

        Args:
            X (np.ndarray): The input feature data.
            y (structured np.ndarray): A structured array from scikit-survival containing
                                       the event indicator and time to event.
        """
        X = torch.tensor(X, dtype=torch.float32)
        D, T = map(np.array, zip(*y))
        D = torch.tensor(D, dtype=torch.float32)
        T = torch.tensor(T, dtype=torch.float32)

        self._init_model(X, D, T) # initializes the model

        dataset = torch.utils.data.TensorDataset(X, T, D) # creates a dataset of the data, time, and event indicator

        # Create generators for reproducibility
        if self.random_state is not None:
            sampler_generator = torch.Generator().manual_seed(self.random_state)
            loader_generator = torch.Generator().manual_seed(self.random_state + 1)
        else:
            sampler_generator = None
            loader_generator = None

        random_sampler = torch.utils.data.RandomSampler(dataset, replacement=False,
                                                        num_samples=self.batch_size - (X.size(0) % self.batch_size),
                                                        generator=sampler_generator)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                 generator=loader_generator)
        
        # The optimizer updates the learnable parameters w and omega
        # A smaller lr is often used for the relevance matrix for more stable updates.
        optimizer_w = torch.optim.Adam([
            {'params': self.w}, # prototypes
            {'params': self.omega, 'lr': self.lr * 0.1}], lr=self.lr) # relevance matrix

        # ---------- Training loop ----------
        for epoch in range(self.epochs):
            # At the start of each epoch, we recalculate the prototype labels
            # based on their updated positions.
            self.fit_labels()
            minibatch_loss = []
            for batch in dataloader:
                x_batch, t_batch, d_batch = batch

                # fill up the batch if not full:
                if x_batch.size(0) < self.batch_size:
                    ids = list(random_sampler)
                    x_batch = torch.cat([x_batch, X[ids]])
                    t_batch = torch.cat([t_batch, T[ids]])
                    d_batch = torch.cat([d_batch, D[ids]])

                optimizer_w.zero_grad()  # Reset gradients
                batch_loss = self.loss_brier(x_batch, t_batch, d_batch)
                minibatch_loss.append(batch_loss)
                batch_loss.backward()  # backpropagate the loss
                optimizer_w.step()     # update weights
                self.normalize_trace() # regularization

            if self.verbose:
                epoch_loss = torch.tensor(minibatch_loss).mean()
                print(f"Epoch: {epoch + 1} / {self.epochs} | Loss: {epoch_loss:.6f}")
        return self


    # Function to visualize specific aspects of the model.
    def vis(self, X, D, T, print_variance_covered=True, true_eigen=False):
        X = torch.tensor(X, dtype=torch.float32)

        ##
        ## projection plot (scatter)
        ##
        plt.figure()
        with torch.no_grad():
            v, u = torch.linalg.eig(self.lambda_mat())
            v, u = torch.real(v), torch.real(u)
            idx = v.argsort(descending=True)
            scale = torch.sqrt(v[idx][:2])

            eig_vec_scaled = scale * u[:, idx][:, :2]

            if self.eig_vec == None:
                self.eig_vec = eig_vec_scaled
            else:
                cosA = torch.nn.functional.cosine_similarity(self.eig_vec, eig_vec_scaled[:, 0, None], dim=0)
                cosB = torch.nn.functional.cosine_similarity(self.eig_vec, eig_vec_scaled[:, 1, None], dim=0)
                i_max_sim = torch.abs(torch.cat([cosA, cosB])).argmax()

                if i_max_sim == 0 or i_max_sim == 3:
                    self.eig_vec[:, 0] = eig_vec_scaled[:, 0] * torch.sign(cosA[0])
                    self.eig_vec[:, 1] = eig_vec_scaled[:, 1] * torch.sign(cosB[1])
                else:
                    self.eig_vec[:, 0] = eig_vec_scaled[:, 1] * torch.sign(cosA[1])
                    self.eig_vec[:, 1] = eig_vec_scaled[:, 0] * torch.sign(cosB[0])

            if true_eigen:
                x_proj = X @ eig_vec_scaled
                w_proj = self.w.detach() @ eig_vec_scaled
            else:
                x_proj = X @ self.eig_vec.cpu()
                w_proj = self.w.detach().cpu() @ self.eig_vec.cpu()

        plt.style.use('default')

        observed = D.astype('?')

        linestyles = ['solid', 'dashdot', 'dashed']

        # datapoints
        fig = plt.scatter(x_proj[observed, 0].numpy(), x_proj[observed, 1].numpy(), c=T[observed], s=75, alpha=0.7)
        cbar = plt.colorbar(fig)
        cbar.set_label('time (days)', rotation=270, labelpad=15)
        plt.scatter(x_proj[~observed, 0].numpy(), x_proj[~observed, 1].numpy(), c='white', s=75, edgecolors='black',
                    alpha=0.3)

        # prototypes
        for i in range(w_proj.size(0)):
            plt.scatter(w_proj[i, 0].numpy(), w_proj[i, 1].numpy(), marker='o', c='white', edgecolor='black', ls=linestyles[i], s=400, linewidth=3)
            proto_txt = (np.arange(w_proj.size(0)) + 1).astype('U')
            plt.text(w_proj[i, 0].numpy(), w_proj[i, 1].numpy(), proto_txt[i],
                     dict(ha='center', va='center_baseline', fontsize=16, color='black', weight='bold'))

        plt.axis('equal')
        plt.xlabel("projection on 1st eigenv. of  $\lambda$")
        plt.ylabel("projection on 2nd eigenv. of  $\lambda$")

        ##
        ## label plots
        ##
        ax2 = plt.figure()
        ax2.clear()
        for i in range(w_proj.size(0)):
            plt.step(self.timepoints.numpy(), self.c[i, :].detach().numpy(), ls=linestyles[i], c='black', where="post", lw=2, label=str(i + 1))
        plt.legend(title='prototype nr')

        plt.xlabel("time (days)")
        plt.ylabel("survival probability S(t)")

        if print_variance_covered:
            print('variance coverd by projection:',
                  v[idx][:2].sum() / v.sum() * 100)

        ##
        ## relevance matrix plot
        ##
        rel_mat = self.lambda_mat().detach().numpy()
        fig = plt.matshow(rel_mat)
        cbar = plt.colorbar(fig)
        cbar.set_label('relevance', rotation=270)
        fig.axes.set_xticks(range(rel_mat.shape[1]))
        fig.axes.set_yticks(range(rel_mat.shape[0]))
        plt.xlabel("feature number")
        plt.ylabel("feature number")

        ##
        ## feature relevance plot
        ##
        plt.figure()
        fig = plt.bar(range(rel_mat.shape[1]), np.diag(rel_mat))
        plt.xticks(range(rel_mat.shape[1]))
        plt.xlabel('Feature number')
        plt.ylabel('relevance')

        ##
        ## eigen value  plot
        ##
        plt.figure()
        fig = plt.bar(range(rel_mat.shape[1]), v)
        plt.xticks(range(rel_mat.shape[1]))
        plt.xlabel('index')
        plt.ylabel('eigen value')
        plt.show()