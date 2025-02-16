if __name__ == '__main__':
    import xml.etree.ElementTree as ET


    def remove_sentences_without_opinions(xml_file):
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 找到所有的<sentence>元素
        sentences = root.findall('.//sentence')

        # 为了安全地在遍历时删除元素，创建一个副本列表
        for sentence in sentences:
            # 检查该<sentence>是否包含<Opinion>子元素
            if not sentence.findall('Opinion'):
                # 如果不包含<Opinion>，则从其父元素中删除该<sentence>
                sentence_parent = sentence.find('..')
                if sentence_parent is not None:  # 确保父元素存在
                    sentence_parent.remove(sentence)

        # 将修改后的XML内容保存到文件
        tree.write("C:/Users/Eric/PycharmProjects/ELLAM/Data_Directory/abc.xml")


    # 替换为你的XML文件名
    xml_file = 'C:/Users/Eric/PycharmProjects/ELLAM/Data_Directory/ABSA16_Restaurants_Train_SB1_v2.xml'
    remove_sentences_without_opinions(xml_file)
