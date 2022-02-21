class Dict2Obj(object):
    """将一个字典转换为类"""

    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        """print 或str 时，让实例对象以字符串格式输出"""
        return "<Dict2Obj: %s>" % self.__dict__