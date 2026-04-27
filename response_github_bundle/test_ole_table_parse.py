"""DOC 파일 OLE 기반 표 구조 파싱 테스트"""

import os
import re
import struct

import olefile

path = 'outputs/ui_runs/review-20260414-145202/source/금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc'


def reconstruct_text(path):
    ole = olefile.OleFileIO(path)
    wd = ole.openstream('WordDocument').read()

    flags = struct.unpack('<H', wd[0x0A:0x0C])[0]
    table_stream_name = '1Table' if (flags & 0x0200) else '0Table'
    table_stream = ole.openstream(table_stream_name).read()

    fcClx = struct.unpack('<I', wd[0x01A2:0x01A6])[0]
    lcbClx = struct.unpack('<I', wd[0x01A6:0x01AA])[0]
    clx_data = table_stream[fcClx:fcClx + lcbClx]

    pos = 0
    full_text = ""
    while pos < len(clx_data):
        t = clx_data[pos]
        if t == 0x01:
            cb = struct.unpack('<H', clx_data[pos + 1:pos + 3])[0]
            pos += 3 + cb
        elif t == 0x02:
            lcb = struct.unpack('<I', clx_data[pos + 1:pos + 5])[0]
            pcd_data = clx_data[pos + 5:pos + 5 + lcb]
            n = (lcb - 4) // 12
            parts = []
            for i in range(n):
                cp_start = struct.unpack('<I', pcd_data[i * 4:i * 4 + 4])[0]
                cp_end = struct.unpack('<I', pcd_data[(i + 1) * 4:(i + 1) * 4 + 4])[0]
                pcd_offset = (n + 1) * 4 + i * 8
                fc_compressed = struct.unpack('<I', pcd_data[pcd_offset + 2:pcd_offset + 6])[0]
                is_unicode = not bool(fc_compressed & 0x40000000)
                fc = fc_compressed & 0x3FFFFFFF
                char_count = cp_end - cp_start
                if is_unicode:
                    text_bytes = wd[fc:fc + char_count * 2]
                    text = text_bytes.decode('utf-16le', errors='replace')
                else:
                    fc = fc // 2
                    text_bytes = wd[fc:fc + char_count]
                    text = text_bytes.decode('cp1252', errors='replace')
                parts.append(text)
            full_text = ''.join(parts)
            break
        else:
            break

    ole.close()
    return full_text


def parse_tables(full_text):
    CELL = chr(7)
    PARA = chr(13)

    current_text = []
    current_row_cells = []
    all_tables = []
    current_table_rows = []
    non_table_buffer = []

    i = 0
    text_len = len(full_text)

    while i < text_len:
        ch = full_text[i]

        if ch == CELL:
            cell_content = ''.join(current_text).strip()
            cell_content = cell_content.replace(PARA, ' ').strip()
            cell_content = re.sub(r'\s*SHAPE\s+\\?\*?\s*MERGEFORMAT\s*', '', cell_content).strip()
            cell_content = re.sub(r'\s+', ' ', cell_content)
            current_row_cells.append(cell_content)
            current_text = []

            if i + 1 < text_len and full_text[i + 1] == PARA:
                if current_row_cells:
                    current_table_rows.append(list(current_row_cells))
                current_row_cells = []
                i += 2
                continue
        elif ch == PARA:
            if not current_row_cells and not current_text:
                i += 1
                continue

            if current_row_cells:
                current_text.append(' ')
            else:
                para_text = ''.join(current_text).strip()
                para_text = re.sub(r'\s*SHAPE\s+\\?\*?\s*MERGEFORMAT\s*', '', para_text).strip()
                if para_text and len(para_text) > 1:
                    if current_table_rows:
                        all_tables.append(current_table_rows)
                        current_table_rows = []
                    non_table_buffer.append(para_text)
                current_text = []
        else:
            current_text.append(ch)

        i += 1

    if current_table_rows:
        all_tables.append(current_table_rows)

    return all_tables, non_table_buffer


def reshape_single_row_table(flat_cells):
    """Single row with many columns -> try to find the right column count."""
    total = len(flat_cells)

    # Try to detect header pattern by looking for empty separator cells
    # Financial tables often have empty cells as group separators
    # Try column counts that divide evenly
    candidates = []
    for ncols in range(5, 25):
        if total % ncols == 0:
            nrows = total // ncols
            if nrows >= 2:
                candidates.append((ncols, nrows))

    if not candidates:
        # Try non-exact divisions (some cells might be merged)
        return None

    # Prefer column counts that give reasonable row counts
    # and where the first row looks like a header
    for ncols, nrows in candidates:
        rows = [flat_cells[i:i + ncols] for i in range(0, total, ncols)]
        # Check first row for header-like content
        first_row = rows[0]
        has_header_keywords = any(
            kw in ' '.join(first_row)
            for kw in ['구분', '구 분', '합계', '합 계', '증감', '금액', '비중', '건수']
        )
        if has_header_keywords:
            return rows

    # Fall back to first candidate
    ncols, nrows = candidates[0]
    return [flat_cells[i:i + ncols] for i in range(0, total, ncols)]


def render_table_markdown(rows, table_idx):
    if not rows:
        return ""

    lines = []
    lines.append(f"### 표 {table_idx}")
    lines.append("")

    max_cols = max(len(r) for r in rows)

    padded = []
    for r in rows:
        padded.append(r + [''] * (max_cols - len(r)))

    # If single row with many columns, try to reshape
    if len(padded) == 1 and len(padded[0]) > 20:
        reshaped = reshape_single_row_table(padded[0])
        if reshaped:
            padded = reshaped
            max_cols = max(len(r) for r in padded)
            padded = [r + [''] * (max_cols - len(r)) for r in padded]

    # Render header
    header = padded[0]
    lines.append("| " + " | ".join(c if c else " " for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in padded[1:]:
        lines.append("| " + " | ".join(c if c else " " for c in row) + " |")

    lines.append("")
    return "\n".join(lines)


def main():
    print("=== DOC OLE 기반 표 파싱 테스트 ===\n")

    full_text = reconstruct_text(path)
    print(f"텍스트 재구성 완료: {len(full_text)} chars")

    all_tables, non_table_buffer = parse_tables(full_text)
    print(f"추출된 표: {len(all_tables)}개")
    print(f"비표 문단: {len(non_table_buffer)}개\n")

    # Build markdown
    md_parts = []
    md_parts.append("# DOC 파일 OLE 기반 파싱 결과")
    md_parts.append("")
    md_parts.append("원본: 금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc")
    md_parts.append("파싱 방식: OLE (olefile) - Piece Table 기반 텍스트 재구성")
    md_parts.append(f"추출된 표: {len(all_tables)}개")
    md_parts.append("")

    md_parts.append("---")
    md_parts.append("")
    md_parts.append("## 본문 텍스트 (표 외)")
    md_parts.append("")
    for p in non_table_buffer:
        if len(p) > 3:
            md_parts.append(p)
            md_parts.append("")

    md_parts.append("---")
    md_parts.append("")
    md_parts.append("## 표 구조")
    md_parts.append("")

    for ti, table in enumerate(all_tables, 1):
        md = render_table_markdown(table, ti)
        if md:
            md_parts.append(md)

    result = "\n".join(md_parts)

    out_path = 'outputs/layout_md/doc_ole_table_parse_result.md'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"결과 저장: {out_path}")
    print(f"\n{'='*60}")
    print("=== 결과 미리보기 ===")
    print(f"{'='*60}\n")
    print(result)


if __name__ == '__main__':
    main()
