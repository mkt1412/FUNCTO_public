from PIL import Image
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import os
import sys
import re
import time
from PIL import Image
from cloud_services.apis.owlv2 import OWLViT, visualize_image
from cloud_services.apis.sam import SAM, visualize_image
from cloud_services.apis.language_model import GPT4V
from cloud_services.image_utils import annotate_masks
from collections import Counter
import ast

@dataclass
class SceneNode:
    name: str
    object_type: Optional[str]
    predicates: List[str] = field(default_factory=list)
    mask: Optional[Any] = None

    def __str__(self):
        return f"( {self.name}, {', '.join(self.predicates)})"



@dataclass
class SceneEdge:
    relationship: str
    source: str
    destination: str


@dataclass
class SceneGraphData:
    nodes: List[SceneNode] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)

    def add_node(self, node: SceneNode):
        self.nodes.append(node)

    def add_edge(self, edge: SceneEdge):
        self.edges.append(edge)


class SceneGraph:
    def __init__(self):
        self.detector = OWLViT()
        self.sam = SAM()
        self.gpt4v = GPT4V()

    @staticmethod
    def get_scene_graph_prompt(predicate: str) -> Tuple[str, str]:
        prompt = f"The additional predicate is [{predicate}]"
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
    def get_alignments_prompt(nodes: str) -> str:
        prompt = f'''Align the number in the marked image with the following nodes: 
{nodes}

with the format:
[
(number, (nodes)),
xxxx
]
        '''
        return prompt

    @staticmethod
    def parse_scene_graph(scene_graph: str) -> Tuple[List[SceneNode], List[SceneEdge], List[str]]:
        node_pattern = re.compile(r"\('([a-zA-Z_]+\d+)', (?:'([^']+)'|None)\)")
        edge_pattern = re.compile(r"\('([a-zA-Z]+)', '([a-zA-Z_]+\d+)', '([a-zA-Z_]+\d+)'\)")

        nodes = [SceneNode(name=obj, object_type=re.sub(r'\d+', '', obj), predicates=ot.split(', ') if ot else []) 
                 for obj, ot in node_pattern.findall(scene_graph)]
        edges = [SceneEdge(relationship=rel, source=src, destination=dest) 
                 for rel, src, dest in edge_pattern.findall(scene_graph)]

        filtered_nodes = [node for node in nodes if not ('table' in node.name.lower() or 'chair' in node.name.lower())]
        filtered_edges = [edge for edge in edges if not any('table' in element or 'chair' in element for element in [edge.source, edge.destination])]

        object_classes = list(set(re.sub(r'\d+', '', node.name) for node in filtered_nodes))

        return filtered_nodes, filtered_edges, object_classes

    def parse_alignments(self, alignments_string: str):
    
        # Convert the string to a Python list of tuples using ast.literal_eval for safety
        data_list = ast.literal_eval(alignments_string)

        # Extract the match relation as (number, name)
        match_relation = [(number, name) for number, (name, _) in data_list]
        return match_relation

    def get_scene_graph_string(self, image: Image.Image, predicate: str) -> str:
        prompt, meta_prompt = self.get_scene_graph_prompt(predicate)
        scene_graph_string = self.gpt4v.chat(prompt, image, meta_prompt=meta_prompt)
        return scene_graph_string

    def get_alignments_string(self, image: Image.Image, nodes: str, scene_graph_string: str) -> str:
        prompt = self.get_alignments_prompt(nodes)
        alignments_string = self.gpt4v.chat(prompt, image, scene_graph_string)
        return alignments_string

    def detect_and_segment(self, image: Image.Image, object_classes: List[str]) -> Tuple[List[Any], List[str]]:
        detected_objects = self.detector.detect_objects(
            image=image,
            text_queries=object_classes,
            bbox_score_top_k=20,
            bbox_conf_threshold=0.15
        )
        best_boxes = {}
        multiple_box_names = []

        for det in detected_objects:
            box_name = det["box_name"]

            if box_name not in best_boxes or det["score"] > best_boxes[box_name]["score"]:
                best_boxes[box_name] = det


        missing_objects = set(object_classes) - set(best_boxes.keys())
        if missing_objects:
            print(f"Missing objects that were not detected or had no best box: {', '.join(missing_objects)}")

        masks = self.sam.segment_by_bboxes(image=image, bboxes=[obj['bbox'] for obj in detected_objects])

        box_names = [obj["box_name"] for obj in detected_objects]

        # Count occurrences of each box_name
        box_count = Counter(box_names)

        # Filter out box_names that appear more than once
        multiple_box_names = [name for name, count in box_count.items() if count > 1]
        return masks, box_names, multiple_box_names

    def get_annotated_segmentation(self, image: Image.Image, masks: List[Any]) -> Image.Image:
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

    def assign_masks_to_nodes(self, image, scene_graph_data: SceneGraphData, masks: List[Any], box_names: List[str], multiple_box_names: List[str], scene_graph_string):
        # print("multiple names:", multiple_box_names)
        masks_muiltiple = []
        # Adding masks to the nodes
        for mask, box_name in zip(masks, box_names):
            if box_name not in multiple_box_names:
                print(f"Box name {box_name} not found in multiple_box_names")
                for node in scene_graph_data.nodes:
                    # Check if the box_name matches the object_type of the node
                    if box_name == node.object_type:
                        node.mask = mask["segmentation"]
                        print(f"Added mask to node: {node.name}")
            else:
                masks_muiltiple.append(mask)
        

        unmatched_nodes = [str(node) for node in scene_graph_data.nodes if node.object_type in multiple_box_names]  # Using the __str__ method of each node
        print("Unmatched nodes:", unmatched_nodes)

        annotated_img = self.get_annotated_segmentation(image, masks)

        if len(unmatched_nodes):
            annotated_img_mutiple = self.get_annotated_segmentation(image, masks_muiltiple)

            get_alignments_string = self.get_alignments_string(annotated_img_mutiple, str(unmatched_nodes), scene_graph_string)    
            parse_alignments = self.parse_alignments(get_alignments_string)
            # for 
            print("alignments:", parse_alignments)
            
            for pair in parse_alignments:
                number, name = pair
                for node in scene_graph_data.nodes:
                    if name == node.name:
                        node.mask = masks[number]["segmentation"]
        return scene_graph_data, annotated_img

if __name__ == "__main__":
    scene_graph_processor = SceneGraph()
    file_path = "cloud_services/images/1.png"
    image = Image.open(file_path)
    predicate = ['Fruits', 'wood', 'with_mark']

    # scene_graph_string = scene_graph_processor.get_scene_graph_string(image, predicate)
    # Simulated scene graph string from GPT-4V output
    scene_graph_string ='''
    nodes=
    [
    ('orange1', 'fruits'),
    ('orange2', 'fruits'),
    ('orange3', 'fruits'),
    ('orange4', 'fruits, with_sticker'),
    ('bowl1', 'wood'),
    ]

    edges=
    [
    ('ON', 'orange1', 'table'),
    ('ON', 'orange2', 'table'),
    ('ON', 'orange3', 'table'),
    ('ON', 'orange4', 'table'),
    ('ON', 'bowl1', 'table'),
    ]
    '''

    nodes, edges, object_classes = scene_graph_processor.parse_scene_graph(scene_graph_string)
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Object classes:", object_classes)

    # Creating a scene graph data structure
    scene_graph_data = SceneGraphData()
    for node in nodes:
        scene_graph_data.add_node(node)
    for edge in edges:
        scene_graph_data.add_edge(edge)

    masks, box_names, multiple_box_names = scene_graph_processor.detect_and_segment(image, object_classes)

    ## Adding masks to the nodes

    scene_graph_data, annotated_img = scene_graph_processor.assign_masks_to_nodes(image, scene_graph_data, masks, box_names, multiple_box_names, scene_graph_string)
    # print("multiple names:", multiple_box_names)
    # masks_muiltiple = []
    # # Adding masks to the nodes
    # for mask, box_name in zip(masks, box_names):
    #     if box_name not in multiple_box_names:
    #         print(f"Box name {box_name} not found in multiple_box_names")
    #         for node in scene_graph_data.nodes:
    #             # Check if the box_name matches the object_type of the node
    #             if box_name == node.object_type:
    #                 node.mask = mask["segmentation"]
    #                 print(f"Added mask to node: {node.name}")
    #     else:
    #         masks_muiltiple.append(mask)
    

    # unmatched_nodes = [str(node) for node in scene_graph_data.nodes if node.object_type in multiple_box_names]  # Using the __str__ method of each node
    # print("Unmatched nodes:", unmatched_nodes)
    
    # annotated_img = scene_graph_processor.get_annotated_segmentation(image, masks_muiltiple)

    # get_alignments_string = scene_graph_processor.get_alignments_string(annotated_img, str(unmatched_nodes), scene_graph_string)    
    # parse_alignments = scene_graph_processor.parse_alignments(get_alignments_string)
    # # for 
    # print("alignments:", parse_alignments)
    
    # for pair in parse_alignments:
    #     number, name = pair
    #     for node in scene_graph_data.nodes:
    #         if name == node.name:
    #             node.mask = masks[number]["segmentation"]

    print(scene_graph_data.nodes)
    # annotated_img.show()
