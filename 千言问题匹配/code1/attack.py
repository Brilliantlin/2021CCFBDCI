# import paddle
# class FGM():
#     def __init__(self, model):
#         self.model = model
#         self.backup = {}
#
#     def attack(self, epsilon=0.3, emb_name='word_embeddings.'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if not param.stop_gradient and emb_name in name:
#                 self.backup[name] = param.clone()
#                 norm = paddle.norm(paddle.to_tensor(param.grad))
#                 if norm != 0:
#                     r_at = epsilon * paddle.to_tensor(param.grad) / norm
#                     param.add(r_at)
#
#     def restore(self, emb_name='word_embeddings.'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if not param.stop_gradient and emb_name in name:
#                 assert name in self.backup
#                 param = self.backup[name]
#         self.backup = {}