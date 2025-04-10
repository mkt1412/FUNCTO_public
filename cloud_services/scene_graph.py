import os
import sys
import re
import time
from PIL import Image
from cloud_services.apis.owlv2 import OWLViT, visualize_image
from cloud_services.apis.sam import SAM, visualize_image
from cloud_services.apis.language_model import GPT4V
from cloud_services.image_utils import annotate_masks


class SceneGraph:
    def __init__(self):
        self.detector = OWLViT()
        self.sam = SAM()
        self.gpt4v = GPT4V()

    @staticmethod
    def get_scene_graph_prompt(predicate):
        prompt = f'''The additional predicate is [{predicate}]'''
        meta_prompt = '''Generate the Scene Graph with the following format, with the given predicate to check if the object satisfies this. Where the node defines the objects and edge defines spatial relationship ['ON', 'INSIDE']. If the predicate is None, don't generate predicate.

nodes=
[
('apple1', 'predicate'),
('flatmat1', 'predicate'),
...
]

edges=
[
('ON', 'apple1', 'mat1'),
...
]


        '''
        return prompt, meta_prompt

    @staticmethod
    def parse_scene_graph(scene_graph):
        # Define patterns for nodes and edges
        node_pattern = re.compile(r"\('([a-zA-Z_]+\d+)', (?:'([^']+)'|None)\)")
        edge_pattern = re.compile(r"\('([a-zA-Z]+)', '([a-zA-Z_]+\d+)', '([a-zA-Z_]+\d+)'\)")

        # Find all nodes and edges using regex
        nodes = node_pattern.findall(scene_graph)
        edges = edge_pattern.findall(scene_graph)

        # Filter nodes that include 'table' or 'chair' and collect remaining nodes
        filtered_nodes = [node for node in nodes if not ('table' in node[0].lower() or 'chair' in node[0].lower())]

        # Filter edges, ensuring neither source nor destination is a filtered-out object
        filtered_edges = [edge for edge in edges if not any('table' in element or 'chair' in element for element in edge)]

        # Use a set to collect unique object classes from filtered nodes, removing numbers
        object_classes_set = set(re.sub(r'\d+', '', obj) for obj, _ in filtered_nodes)

        # Convert set to list to finalize the object classes
        object_classes = list(object_classes_set)

        return filtered_nodes, filtered_edges, object_classes

    def get_scene_graph_string(self, image, predicate):
        prompt, meta_prompt = self.get_scene_graph_prompt(predicate)
        scene_graph_string = self.gpt4v.chat(prompt, image, meta_prompt=meta_prompt)

        return scene_graph_string
    
    # def extact_scene_graph(self, scene_graph_string)
    #     nodes, edges, object_classes = self.parse_scene_graph(scene_graph_string)
    #     return nodes, edges, object_classes, image

    def detect_and_segment(self, image, object_classes):
        detected_objects = self.detector.detect_objects(
            image=image,
            text_queries=object_classes,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.15
        )
        best_boxes = {}
        for det in detected_objects:
            box_name = det["box_name"]
            if box_name not in best_boxes or det["score"] > best_boxes[box_name]["score"]:
                best_boxes[box_name] = det

        missing_objects = set(object_classes) - set(best_boxes.keys())
        if missing_objects:
            print(f"Missing objects that were not detected or had no best box: {', '.join(missing_objects)}")
        # missing_objects
        masks = self.sam.segment_by_bboxes(image=image, bboxes=[obj['bbox'] for obj in detected_objects])
        
        box_names = [obj["box_name"] for obj in detected_objects]
        return masks, box_names
    
    def get_annotated_segmentation(self, image, masks):
        segment_img = annotate_masks(
            image, 
            masks=[anno["segmentation"] for anno in masks],
            label_mode="1",
            alpha=0.05,
            draw_mask=True, 
            draw_mark=True, 
            draw_box=True,
            mark_position='top_left'
        )
        return segment_img

if __name__ == "__main__":
    scene_graph_processor = SceneGraph()
    file_path = "cloud_services/images/1.png"
    image = Image.open(file_path)
    predicate = "Fruits"
    # scene_graph_string = scene_graph_processor.get_scene_graph_string(image, predicate)
    # print(scene_graph_string)
    scene_graph_string ='''
    nodes=
    [
    ('orange1', 'fruits'),
    ('orange2', 'fruits'),
    ('orange3', 'fruits'),
    ('orange4', 'fruits, with_mark'),
    ('bowl1', 'wood'),
    ('book1', None),
    ('pen1', None),
    ('cable1', None)
    ]

    edges=
    [
    ('ON', 'orange1', 'table'),
    ('ON', 'orange2', 'table'),
    ('ON', 'orange3', 'table'),
    ('ON', 'orange4', 'table'),
    ('ON', 'bowl1', 'table'),
    ('ON', 'book1', 'table'),
    ('ON', 'pen1', 'table'),
    ('ON', 'cable1', 'table')
    ]
    '''
    nodes, edges, object_classes = scene_graph_processor.parse_scene_graph(scene_graph_string)

    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Object classes:", object_classes)
    masks, box_names = scene_graph_processor.detect_and_segment(image, object_classes)
    print("Box names:", box_names)

    annotated_img = scene_graph_processor.get_annotated_segmentation(image, masks)
    annotated_img.show()