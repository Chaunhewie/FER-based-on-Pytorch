import torch


class DataPrefetcher():
    """
    ���ڽ�loader��һ����װ��ʹ��GPU�������ݵ�Ԥ��ȡ�ٶȡ�
    �������https://zhuanlan.zhihu.com/p/66145913?utm_source=qq&utm_medium=social&utm_oi=984715653122629632

    ʾ����
    >>>training_data_loader = DataLoader(
    >>>    dataset=train_dataset,
    >>>    num_workers=opts.threads,
    >>>    batch_size=opts.batchSize,
    >>>    pin_memory=True,
    >>>    shuffle=True,
    >>>)
    >>>for iteration, batch in enumerate(training_data_loader, 1):
    >>>    # ѵ������
    >>>
    >>>#-------------������---------
    >>>
    >>>data, label = prefetcher.next()
    >>>iteration = 0
    >>>while data is not None:
    >>>    iteration += 1
    >>>    # ѵ������
    >>>    data, label = prefetcher.next()

    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
