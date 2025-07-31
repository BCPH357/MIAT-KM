#!/usr/bin/env python3
import os
import time
import argparse
from triple_extraction import process_pdf_directory, save_triples_to_csv
from import_to_neo4j import wait_for_neo4j, Neo4jImporter

def main():
    parser = argparse.ArgumentParser(description='處理 PDF 文件並將三元組導入到 Neo4j')
    parser.add_argument('--pdf-dir', default='/app/data/pdf', help='PDF 文件目錄')
    parser.add_argument('--output-file', default='/app/data/processed/triples.csv', help='輸出 CSV 文件路徑')
    parser.add_argument('--neo4j-uri', default='bolt://neo4j:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j 用戶名')
    parser.add_argument('--neo4j-password', default='password123', help='Neo4j 密碼')
    args = parser.parse_args()

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 檢查 PDF 目錄
    if not os.path.exists(args.pdf_dir):
        print(f"PDF 目錄不存在: {args.pdf_dir}")
        return
    
    pdf_files = [f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"PDF 目錄中沒有 PDF 文件: {args.pdf_dir}")
        return
    
    print(f"找到 {len(pdf_files)} 個 PDF 文件")
    
    # 處理 PDF 文件並提取三元組
    print("開始處理 PDF 文件...")
    triples = process_pdf_directory(args.pdf_dir)
    
    # 保存三元組到 CSV 文件
    save_triples_to_csv(triples, args.output_file)
    print(f"共提取了 {len(triples)} 個三元組，已保存到 {args.output_file}")
    
    # 等待 Neo4j 服務啟動
    if not wait_for_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password):
        print("無法連接到 Neo4j，退出程序")
        return
    
    # 導入三元組到 Neo4j
    print("開始導入三元組到 Neo4j...")
    importer = Neo4jImporter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        importer.import_triples(args.output_file)
        
        # 計算導入的節點和關係數量
        node_count, rel_count = importer.count_nodes_and_relationships()
        print(f"成功導入到 Neo4j: {node_count} 個節點和 {rel_count} 個關係")
    finally:
        importer.close()

if __name__ == "__main__":
    main() 