import torch
from models.modules import PreTrained_Model

def relaxed_distortion_measure(func, z, eta=0.2, metric='identity', create_graph=True):
    if metric == 'identity':
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=create_graph)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()
        return TrG2/TrG**2

    elif isinstance(metric, PreTrained_Model):

        model = metric.get_model()
        # model = metric.class_name()
        # model.load_state_dict(torch.load(metric.parameter_path))
        model.to(z)
        
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z

        def comp_func(x):
            return model(func(x))

        v = torch.randn(z.size()).to(z)

        Jv = torch.autograd.functional.jvp(comp_func, z_augmented, v=v, create_graph=create_graph)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(comp_func, z_augmented, v=Jv, create_graph=create_graph)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()

        # Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=create_graph)[1]
        # HJv = torch.autograd.functional.jvp(model, func(z_augmented), v=Jv, create_graph=create_graph)[1]

        # TrG = torch.sum(HJv.view(bs, -1)**2, dim=1).mean()

        # HTHJv = (torch.autograd.functional.vjp(model, func(z_augmented), v=HJv, create_graph=create_graph)[1]).view(bs, -1)
        # JTHTHJv = (torch.autograd.functional.vjp(func, z_augmented, v=HTHJv.view(bs, Jv.shape[1], Jv.shape[2], Jv.shape[3]), create_graph=create_graph)[1]).view(bs, -1)

        # TrG2 = torch.sum(JTHTHJv**2, dim=1).mean()

        return TrG2/TrG**2

    else:
        raise NotImplementedError


def get_flattening_scores(G, mode='condition_number'):
    if mode == 'condition_number':
        S = torch.svd(G).S
        scores = S.max(1).values/S.min(1).values
    elif mode == 'variance':
        G_mean = torch.mean(G, dim=0, keepdim=True)
        A = torch.inverse(G_mean)@G
        scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
    else:
        pass
    return scores

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac 

def get_pullbacked_Riemannian_metric(func, z):
    J = jacobian_decoder_jvp_parallel(func, z, v=None)
    G = torch.einsum('nij,nik->njk', J, J)
    return G

# def get_doubled_pullbacked_Riemannian_metric(func2, func, z):
#     bs = z.shape[0]
#     J = jacobian_decoder_jvp_parallel(func, z, v=None)
#     H = jacobian_decoder_jvp_parallel(func2, func(z).view(bs, -1), v=None)
#     HJ = torch.einsum('nij, njk -> nik', H, J)
#     G = torch.einsum('nij,nik->njk', HJ, HJ)
#     return G

