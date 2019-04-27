#include "data_struct.h"


std::ostream & operator<<(std::ostream & stream, const cooEntry & item) {
    stream << item.row << "\t" << item.col << "\t" << item.val;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const cooEntry_row & item) {
    stream << item.row << "\t" << item.col << "\t" << item.val;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const vecEntry & item) {
    stream << item.row << "\t" << item.val;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const tuple1 & item) {
    stream << item.idx1 << "\t" << item.idx2;
    return stream;
}

std::ostream & operator<<(std::ostream & stream, const vecCol & item) {
    stream << item.rv->row << "\t" << *item.c << "\t" << item.rv->val;
    return stream;
}


bool row_major (const cooEntry& node1, const cooEntry& node2)
{
    if(node1.row < node2.row)
        return (true);
    else if(node1.row == node2.row)
        return(node1.col <= node2.col);
    else
        return false;
}


bool vecCol_col_major (const vecCol& node1, const vecCol& node2)
{
    if(*node1.c < *node2.c)
        return (true);
    else if(*node1.c == *node2.c)
        return((*node1.rv).row <= (*node2.rv).row);
    else
        return false;
}
