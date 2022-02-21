import logging
class setLog:
    def __int__(self):
        self.logger = None

    def setloger(self,filename, cmd_level=logging.DEBUG, filter_level=None, logger_name="app",):

        self.file_name = filename
        self.logger = logging.getLogger(logger_name) #单例，loggname确定唯一实例
        self.logger.setLevel(logging.DEBUG)
        handlers = []
        handlers.append(self.get_file_handler(filename))
        handlers.append(self.get_cmd_handler())

        for handler in handlers:
            # print('append add handler')
            self.logger.addHandler(handler)


    def get_filter(self, level):
        """输出过滤器 只输出你设置好的等级"""
        info_filter = logging.Filter()
        info_filter.filter = lambda record: record.levelno == level
        return info_filter

    def get_file_handler(self,filename):
        format = logging.Formatter(fmt="%(message)s")
        handler = logging.FileHandler(filename,mode='w+')
        filt = self.get_filter(logging.INFO)
        handler.setLevel(logging.INFO)
        handler.setFormatter(format)
        handler.addFilter(filt)
        return handler

    def get_cmd_handler(self):
        formatter = logging.Formatter(
            "%(asctime)s\t[%(filename)s-%(funcName)s]-[line:%(lineno)d]-%(levelname)s:%(message)s")
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        return handler

setlog = setLog()
setlog.logger = None