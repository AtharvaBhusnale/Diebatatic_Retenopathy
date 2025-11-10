import java.util.PriorityQueue;

/* Time complexity:O(nlog n)  Space complexity:O(N) */

class Node implements Comparable<Node> {

    char ch;
    int freq;
    Node left, right;

    public Node(char ch, int freq) {
        this.ch = ch;
        this.freq = freq;
        this.left = null;
        this.right = null;
    }

    // For priority queue sorting (ascending order of frequency)
    public int compareTo(Node other) {
        return this.freq - other.freq;
    }
}

public class HuffmanEncoding {

    // Recursive function to print Huffman codes
    static void printHuffmanCodes(Node root, String code) {
        if (root == null) return;

        // If it's a leaf node → print the character and its code
        if (
            root.left == null &&
            root.right == null &&
            Character.isLetter(root.ch)
        ) {
            System.out.println(root.ch + " : " + code);
            return;
        }

        // Traverse left (append 0) and right (append 1)
        printHuffmanCodes(root.left, code + "0");
        printHuffmanCodes(root.right, code + "1");
    }

    public static void main(String[] args) {
        // ✅ Static Input
        char[] characters = { 'A', 'B', 'C', 'D', 'E', 'F' };
        int[] frequency = { 5, 9, 12, 13, 16, 45 };
        int n = characters.length;

        // Step 1: Create a priority queue (min-heap)
        PriorityQueue<Node> pq = new PriorityQueue<>();

        for (int i = 0; i < n; i++) {
            pq.add(new Node(characters[i], frequency[i]));
        }

        // Step 2: Build Huffman Tree
        while (pq.size() > 1) {
            // Extract two smallest frequency nodes
            Node left = pq.poll();
            Node right = pq.poll();

            // Create new internal node with combined frequency
            Node newNode = new Node('-', left.freq + right.freq);
            newNode.left = left;
            newNode.right = right;

            pq.add(newNode);
        }

        // Step 3: Print Huffman Codes
        Node root = pq.peek();
        System.out.println("Huffman Codes:");
        printHuffmanCodes(root, "");
    }
}
