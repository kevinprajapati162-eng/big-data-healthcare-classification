import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class CSVSorter extends Configured implements Tool {

    // Fixed header: these 12 columns remain the same in every generated file.
    private static final String[] FIXED_HEADER = {
        "PatientNHSNumber", "Name", "Age", "Gender", "Date_of_Admission",
        "Medical_Record_Number", "Medical_History", "Chief_Complaint",
        "History_of_Present_Illness", "Physical_Examination", "Assessment_and_Plan",
        "NHS_Trust_Region"
    };

    // Define fixed output directory
    private static final String OUTPUT_DIR = "/user/hadoop/output/";
    private static final String REGIONS_DIR = "/user/hadoop/output/regions/";
    
    // Enumeration for sort types
    public static enum SortType {
        NHS_NUMBER,
        TRUST_REGION,
        SYMPTOMS,
        REGIONS_SPLIT
    }

    // PatientRecord models one row in the CSV.
    public static class PatientRecord {
        String nhsNumber;
        String name;
        String age;
        String gender;
        String dateOfAdmission;  // Combined if raw has 13 fields.
        String medicalRecordNumber;
        String medicalHistory;
        String chiefComplaint;
        String historyOfPresentIllness;
        String physicalExamination;
        String assessmentAndPlan;
        String nhsTrustRegion;

        // Constructor accepts a raw array of fields.
        // If fields.length == 13 then columns 4 and 5 are merged.
        public PatientRecord(String[] fields) {
            if (fields.length == 13) {
                this.nhsNumber = fields[0];
                this.name = fields[1];
                this.age = fields[2];
                this.gender = fields[3];
                // Combine fields[4] and fields[5] into Date_of_Admission.
                this.dateOfAdmission = fields[4] + ", " + fields[5];
                this.medicalRecordNumber = fields[6];
                this.medicalHistory = fields[7];
                this.chiefComplaint = fields[8];
                this.historyOfPresentIllness = fields[9];
                this.physicalExamination = fields[10];
                this.assessmentAndPlan = fields[11];
                this.nhsTrustRegion = fields[12];
            } else if (fields.length == 12) {
                this.nhsNumber = fields[0];
                this.name = fields[1];
                this.age = fields[2];
                this.gender = fields[3];
                this.dateOfAdmission = fields[4];
                this.medicalRecordNumber = fields[5];
                this.medicalHistory = fields[6];
                this.chiefComplaint = fields[7];
                this.historyOfPresentIllness = fields[8];
                this.physicalExamination = fields[9];
                this.assessmentAndPlan = fields[10];
                this.nhsTrustRegion = fields[11];
            } else {
                throw new IllegalArgumentException("Expected 12 or 13 fields, got " + fields.length);
            }
        }

        // Returns the fields in the fixed order.
        public String[] toArray() {
            return new String[] {
                nhsNumber,
                name,
                age,
                gender,
                dateOfAdmission,
                medicalRecordNumber,
                medicalHistory,
                chiefComplaint,
                historyOfPresentIllness,
                physicalExamination,
                assessmentAndPlan,
                nhsTrustRegion
            };
        }
        
        // Escapes a value for CSV output.
        private String escapeCSV(String value) {
            if (value == null) return "";
            if (value.contains(",") || value.contains("\"")) {
                value = value.replace("\"", "\"\"");
                return "\"" + value + "\"";
            }
            return value;
        }
        
        // Returns a CSV-formatted line for this record.
        public String toCSVLine() {
            String[] arr = this.toArray();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < arr.length; i++) {
                sb.append(escapeCSV(arr[i]));
                if (i < arr.length - 1) {
                    sb.append(",");
                }
            }
            return sb.toString();
        }
        
        // Generate a key for sorting based on sort type
        public String getSortKey(SortType sortType) {
            switch (sortType) {
                case NHS_NUMBER:
                    return nhsNumber + "\t" + dateOfAdmission;
                case TRUST_REGION:
                    return nhsTrustRegion + "\t" + nhsNumber;
                case SYMPTOMS:
                    return chiefComplaint;
                case REGIONS_SPLIT:
                    return normalizeRegionForFilename(nhsTrustRegion) + "\t" + nhsNumber;
                default:
                    return nhsNumber;
            }
        }
        
        // Generate proper output based on sort type
        public String getFormattedOutput(SortType sortType) {
            String[] fields = toArray();
            StringBuilder sb = new StringBuilder();
            
            if (sortType == SortType.NHS_NUMBER) {
                // Standard order: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
                for (int i = 0; i < fields.length; i++) {
                    sb.append(escapeCSV(fields[i]));
                    if (i < fields.length - 1) {
                        sb.append(",");
                    }
                }
            } else if (sortType == SortType.TRUST_REGION) {
                // Order for region sort: 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                int[] order = {11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                for (int i = 0; i < order.length; i++) {
                    sb.append(escapeCSV(fields[order[i]]));
                    if (i < order.length - 1) {
                        sb.append(",");
                    }
                }
            } else if (sortType == SortType.SYMPTOMS) {
                // Order for symptoms sort: 7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11
                int[] order = {7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11};
                for (int i = 0; i < order.length; i++) {
                    sb.append(escapeCSV(fields[order[i]]));
                    if (i < order.length - 1) {
                        sb.append(",");
                    }
                }
            } else if (sortType == SortType.REGIONS_SPLIT) {
                // Standard order for region-specific files
                for (int i = 0; i < fields.length; i++) {
                    sb.append(escapeCSV(fields[i]));
                    if (i < fields.length - 1) {
                        sb.append(",");
                    }
                }
            }
            
            return sb.toString();
        }
    }

    // Helper method to normalize region strings for comparison.
    private static String normalizeRegion(String region) {
        if (region == null) {
            return "";
        }
        return region.toLowerCase().replace("_", " ").trim();
    }
    
    // Helper method to normalize region for filename (replace spaces, special chars)
    private static String normalizeRegionForFilename(String region) {
        if (region == null) {
            return "unknown_region";
        }
        return region.trim()
                .toLowerCase()
                .replace(" ", "_")
                .replace(",", "")
                .replace("&", "and")
                .replaceAll("[^a-z0-9_]", "");
    }
    
    // Parses a CSV line into fields while handling quotes and escaped quotes.
    private static String[] parseCSVLine(String line) {
        List<String> tokens = new ArrayList<>();
        if (line == null || line.isEmpty()) {
            return new String[0];
        }
        StringBuilder token = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        token.append('"');
                        i++; // Skip the escaped quote.
                    } else {
                        inQuotes = false;
                    }
                } else {
                    token.append(c);
                }
            } else {
                if (c == '"') {
                    inQuotes = true;
                } else if (c == ',') {
                    tokens.add(token.toString());
                    token.setLength(0);
                } else {
                    token.append(c);
                }
            }
        }
        tokens.add(token.toString());
        return tokens.toArray(new String[0]);
    }
    
    // MapReduce Mapper class
    public static class PatientMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Text outputKey = new Text();
        private Text outputValue = new Text();
        private SortType sortType;
        private MultipleOutputs<Text, Text> multipleOutputs;
        
        @Override
        protected void setup(Context context) {
            sortType = SortType.valueOf(context.getConfiguration().get("sort.type"));
            if (sortType == SortType.REGIONS_SPLIT) {
                multipleOutputs = new MultipleOutputs<>(context);
            }
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            
            // Skip the header line
            if (line.startsWith("PatientNHSNumber") || line.startsWith("NHS_Trust_Region") || line.startsWith("Chief_Complaint")) {
                return;
            }
            
            String[] fields = parseCSVLine(line);
            
            if (fields.length == 12 || fields.length == 13) {
                try {
                    PatientRecord record = new PatientRecord(fields);
                    
                    if (sortType == SortType.REGIONS_SPLIT) {
                        // For region-specific files, emit to MultipleOutputs
                        String region = normalizeRegionForFilename(record.nhsTrustRegion);
                        if (region.isEmpty()) {
                            region = "unknown_region";
                        }
                        outputKey.set(record.nhsNumber);
                        outputValue.set(record.toCSVLine());
                        multipleOutputs.write("region", outputKey, outputValue, region);
                    } else {
                        // For regular sorting jobs
                        outputKey.set(record.getSortKey(sortType));
                        outputValue.set(record.toCSVLine());
                        context.write(outputKey, outputValue);
                    }
                } catch (IllegalArgumentException e) {
                    // Log and skip problematic records
                    System.err.println("Error processing record: " + e.getMessage());
                }
            }
        }
        
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (multipleOutputs != null) {
                multipleOutputs.close();
            }
        }
    }
    
    // MapReduce Reducer class
    public static class PatientReducer extends Reducer<Text, Text, Text, Text> {
        private SortType sortType;
        private MultipleOutputs<Text, Text> multipleOutputs;
        private Set<String> processedRegions = new HashSet<>();
        
        @Override
        protected void setup(Context context) {
            sortType = SortType.valueOf(context.getConfiguration().get("sort.type"));
            if (sortType == SortType.REGIONS_SPLIT) {
                multipleOutputs = new MultipleOutputs<>(context);
            }
        }
        
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                String line = value.toString();
                String[] fields = parseCSVLine(line);
                
                try {
                    PatientRecord record = new PatientRecord(fields);
                    
                    if (sortType == SortType.REGIONS_SPLIT) {
                        // For region-specific files
                        String region = normalizeRegionForFilename(record.nhsTrustRegion);
                        if (region.isEmpty()) {
                            region = "unknown_region";
                        }
                        
                        if (!processedRegions.contains(region)) {
                            // Write header for this region if we haven't seen it before
                            multipleOutputs.write("region", new Text(""), new Text(String.join(",", FIXED_HEADER)), region);
                            processedRegions.add(region);
                        }
                        
                        multipleOutputs.write("region", new Text(""), new Text(record.getFormattedOutput(sortType)), region);
                    } else {
                        // Write with the appropriate column order based on sort type
                        context.write(new Text(""), new Text(record.getFormattedOutput(sortType)));
                    }
                } catch (IllegalArgumentException e) {
                    System.err.println("Error processing record in reducer: " + e.getMessage());
                }
            }
        }
        
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (multipleOutputs != null) {
                multipleOutputs.close();
            }
        }
    }
    
    // Custom OutputFormat to write the header line
    public static class CSVOutputFormat extends TextOutputFormat<Text, Text> {
        private SortType sortType;
        
        @Override
        public RecordWriter<Text, Text> getRecordWriter(TaskAttemptContext context) throws IOException, InterruptedException {
            sortType = SortType.valueOf(context.getConfiguration().get("sort.type"));
            
            final RecordWriter<Text, Text> originalWriter = super.getRecordWriter(context);
            
            // Determine the header order based on sort type
            String headerLine;
            if (sortType == SortType.NHS_NUMBER) {
                headerLine = String.join(",", FIXED_HEADER);
            } else if (sortType == SortType.TRUST_REGION) {
                int[] order = {11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                String[] orderedHeader = new String[FIXED_HEADER.length];
                for (int i = 0; i < order.length; i++) {
                    orderedHeader[i] = FIXED_HEADER[order[i]];
                }
                headerLine = String.join(",", orderedHeader);
            } else if (sortType == SortType.SYMPTOMS) {
                int[] order = {7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11};
                String[] orderedHeader = new String[FIXED_HEADER.length];
                for (int i = 0; i < order.length; i++) {
                    orderedHeader[i] = FIXED_HEADER[order[i]];
                }
                headerLine = String.join(",", orderedHeader);
            } else { // REGIONS_SPLIT - headers handled by MultipleOutputs
                headerLine = String.join(",", FIXED_HEADER);
            }
            
            // Don't write header for REGIONS_SPLIT as it's handled in the reducer
            if (sortType != SortType.REGIONS_SPLIT) {
                try {
                    // Write the header line first
                    originalWriter.write(new Text(""), new Text(headerLine));
                } catch (InterruptedException e) {
                    throw new IOException("Interrupted while writing header", e);
                }
            }
            
            return new RecordWriter<Text, Text>() {
                @Override
                public void write(Text key, Text value) throws IOException, InterruptedException {
                    originalWriter.write(key, value);
                }
                
                @Override
                public void close(TaskAttemptContext context) throws IOException, InterruptedException {
                    originalWriter.close(context);
                }
            };
        }
    }
    
    // Helper method to run a MapReduce job
    private Job configureAndRunJob(String inputPath, String outputPath, SortType sortType) throws Exception {
        Configuration conf = getConf();
        conf.set("sort.type", sortType.name());
        
        Job job = Job.getInstance(conf, "CSVSorter-" + sortType.name());
        job.setJarByClass(CSVSorter.class);
        
        job.setMapperClass(PatientMapper.class);
        job.setReducerClass(PatientReducer.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(CSVOutputFormat.class);
        
        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        
        // Configure MultipleOutputs for region-specific files
        if (sortType == SortType.REGIONS_SPLIT) {
            MultipleOutputs.addNamedOutput(job, "region", TextOutputFormat.class, Text.class, Text.class);
        }
        
        return job;
    }
    
    @Override
    public int run(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: hadoop jar <jar-file> CSVSorter <hdfs-input-path>");
            System.err.println("Example: hadoop jar myjar.jar CSVSorter /user/hadoop/input/MedicalFiles.csv");
            return 1;
        }
        
        String inputPath = args[0];
        
        // Create output directories if they don't exist
        Configuration conf = getConf();
        FileSystem fs = FileSystem.get(conf);
        Path outputDir = new Path(OUTPUT_DIR);
        Path regionsDir = new Path(REGIONS_DIR);
        
        if (fs.exists(outputDir)) {
            fs.delete(outputDir, true);
        }
        fs.mkdirs(outputDir);
        
        // Run four MapReduce jobs for the different sorts and region-specific files
        Job job1 = configureAndRunJob(inputPath, OUTPUT_DIR + "sorted_by_NHSNumber", SortType.NHS_NUMBER);
        boolean success1 = job1.waitForCompletion(true);
        
        Job job2 = configureAndRunJob(inputPath, OUTPUT_DIR + "sorted_by_TrustRegion", SortType.TRUST_REGION);
        boolean success2 = job2.waitForCompletion(true);
        
        Job job3 = configureAndRunJob(inputPath, OUTPUT_DIR + "sorted_by_Symptoms", SortType.SYMPTOMS);
        boolean success3 = job3.waitForCompletion(true);
        
        // New job for region-specific files
        Job job4 = configureAndRunJob(inputPath, REGIONS_DIR, SortType.REGIONS_SPLIT);
        boolean success4 = job4.waitForCompletion(true);
        
        if (success1 && success2 && success3 && success4) {
            System.out.println("All MapReduce jobs completed successfully!");
            System.out.println("Region-specific files are available in: " + REGIONS_DIR);
            
            // Run the interactive filtering portion outside the MapReduce framework
            runInteractiveFiltering(inputPath);
            
            return 0;
        } else {
            System.err.println("One or more MapReduce jobs failed!");
            return 1;
        }
    }
    
    // The interactive filtering functionality remains similar to the original but reads from HDFS
    private void runInteractiveFiltering(String hdfsInputPath) {
        List<PatientRecord> records = new ArrayList<>();
        
        // Read the CSV file from HDFS
        try {
            Configuration conf = getConf();
            FileSystem fs = FileSystem.get(conf);
            Path inputPath = new Path(hdfsInputPath);
            
            if (!fs.exists(inputPath)) {
                System.err.println("Input file does not exist: " + hdfsInputPath);
                return;
            }
            
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(inputPath)))) {
                // Skip the header line
                String headerLine = br.readLine();
                if (headerLine == null) {
                    System.out.println("Empty CSV file.");
                    return;
                }
                
                String line;
                while ((line = br.readLine()) != null) {
                    String[] fields = parseCSVLine(line);
                    if (fields.length == 13 || fields.length == 12) {
                        records.add(new PatientRecord(fields));
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading HDFS CSV file: " + e.getMessage());
            e.printStackTrace();
            return;
        }
        
        System.out.println("Successfully read " + records.size() + " records from HDFS: " + hdfsInputPath);
        
        // Get all unique regions for displaying options
        Set<String> uniqueRegions = new HashSet<>();
        for (PatientRecord record : records) {
            if (record.nhsTrustRegion != null && !record.nhsTrustRegion.trim().isEmpty()) {
                uniqueRegions.add(record.nhsTrustRegion.trim());
            }
        }
        
        // Display all available regions
        System.out.println("\nAvailable NHS Trust Regions in the dataset:");
        int regionCounter = 1;
        for (String region : uniqueRegions) {
            System.out.println(regionCounter + ": " + region);
            regionCounter++;
        }
        
        // Interactive filtering portion
        Scanner scanner = new Scanner(System.in);
        boolean doMoreFiltering = true;
        while (doMoreFiltering) {
            System.out.println("\nFilter Options:");
            System.out.println("1: Filter by NHS Number (comma separated for multiple)");
            System.out.println("2: Filter by Region (comma separated for multiple)");
            System.out.println("3: Filter by Symptom(s) (comma separated for multiple)");
            System.out.println("4: Filter by Region and Symptom(s) (combined search)");
            System.out.println("5: List all regions and their patient counts");
            System.out.println("6: Exit filtering");
            System.out.print("Enter your choice (1-6): ");
            String filterChoice = scanner.nextLine().trim();
            
            if (filterChoice.equals("6")) {
                System.out.println("Exiting filtered details display.");
                break;
            }
            
            List<PatientRecord> filteredRecords = new ArrayList<>();
            switch (filterChoice) {
                case "1": {
                    System.out.print("Enter desired NHS Number(s) separated by comma: ");
                    String filterValue = scanner.nextLine().trim();
                    String[] nhsNumbers = filterValue.split(",");
                    Set<String> nhsSet = new HashSet<>();
                    for (String num : nhsNumbers) {
                        nhsSet.add(num.trim().toLowerCase());
                    }
                    for (PatientRecord record : records) {
                        if (nhsSet.contains(record.nhsNumber.toLowerCase())) {
                            filteredRecords.add(record);
                        }
                    }
                    break;
                }
                case "2": {
                    System.out.print("Enter desired Region(s) separated by comma: ");
                    String filterValue = scanner.nextLine().trim();
                    String[] regions = filterValue.split(",");
                    Set<String> regionSet = new HashSet<>();
                    for (String reg : regions) {
                        regionSet.add(normalizeRegion(reg));
                    }
                    for (PatientRecord record : records) {
                        String recordRegion = normalizeRegion(record.nhsTrustRegion);
                        if (regionSet.contains(recordRegion)) {
                            filteredRecords.add(record);
                        }
                    }
                    break;
                }
                case "3": {
                    System.out.print("Enter desired Symptom(s) separated by comma: ");
                    String filterValue = scanner.nextLine().trim();
                    String[] symptoms = filterValue.split(",");
                    Set<String> symptomSet = new HashSet<>();
                    for (String symp : symptoms) {
                        symptomSet.add(symp.trim().toLowerCase());
                    }
                    for (PatientRecord record : records) {
                        String symptomData = record.chiefComplaint.toLowerCase();
                        for (String symp : symptomSet) {
                            if (symptomData.contains(symp)) {
                                filteredRecords.add(record);
                                break;
                            }
                        }
                    }
                    break;
                }
                case "4": {
                    System.out.print("Enter desired Region(s) separated by comma: ");
                    String regionInput = scanner.nextLine().trim();
                    System.out.print("Enter desired Symptom(s) separated by comma: ");
                    String symptomInput = scanner.nextLine().trim();
                    String[] regions = regionInput.split(",");
                    Set<String> regionSet = new HashSet<>();
                    for (String reg : regions) {
                        regionSet.add(normalizeRegion(reg));
                    }
                    String[] symptoms = symptomInput.split(",");
                    Set<String> symptomSet = new HashSet<>();
                    for (String symp : symptoms) {
                        symptomSet.add(symp.trim().toLowerCase());
                    }
                    for (PatientRecord record : records) {
                        String recordRegion = normalizeRegion(record.nhsTrustRegion);
                        if (regionSet.contains(recordRegion)) {
                            String symptomData = record.chiefComplaint.toLowerCase();
                            for (String symp : symptomSet) {
                                if (symptomData.contains(symp)) {
                                    filteredRecords.add(record);
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }
                case "5": {
                    // Count patients per region
                    Map<String, Integer> regionCounts = new HashMap<>();
                    for (PatientRecord record : records) {
                        String region = record.nhsTrustRegion;
                        if (region == null || region.trim().isEmpty()) {
                            region = "Unknown";
                        }
                        regionCounts.put(region, regionCounts.getOrDefault(region, 0) + 1);
                    }
                    
                    System.out.println("\nPatient counts by NHS Trust Region:");
                    List<Map.Entry<String, Integer>> sortedEntries = new ArrayList<>(regionCounts.entrySet());
                    sortedEntries.sort((e1, e2) -> e2.getValue().compareTo(e1.getValue())); // Sort by count descending
                    
                    for (Map.Entry<String, Integer> entry : sortedEntries) {
                        System.out.println(entry.getKey() + ": " + entry.getValue() + " patients");
                    }
                    
                    System.out.println("\nRegion-specific CSV files have been generated at: " + REGIONS_DIR);
                    continue;
                }
                default:
                    System.out.println("Invalid choice. Please try again.");
                    continue;
            }
            
            // Display the filtered records
            if (filteredRecords.isEmpty()) {
                System.out.println("No records found matching the filter criteria.");
            } else {
                System.out.println("\nFiltered Patient Details (each field on a new line):");
                System.out.println("Found " + filteredRecords.size() + " matching records.");
                printLabeledRecordsInNewLines(filteredRecords);
            }
            
            // Ask if the user wants to perform more filtering searches.
            System.out.print("Do you want to perform another filtering search? (Y/N): ");
            String response = scanner.nextLine().trim().toLowerCase();
            if (response.startsWith("n")) {
                doMoreFiltering = false;
            }
        }
        scanner.close();
    }
    
    // Prints each filtered patient record with each field on a separate line,
    // and prints a dashed line after each patient.
    private static void printLabeledRecordsInNewLines(List<PatientRecord> records) {
        for (PatientRecord record : records) {
            String[] values = record.toArray();
            for (int i = 0; i < FIXED_HEADER.length; i++) {
                System.out.println(FIXED_HEADER[i] + ": " + values[i]);
            }
            // Print a dashed line to separate records.
            System.out.println("------------------------------------------------------");
        }
    }
    
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new CSVSorter(), args);
        System.exit(exitCode);
    }
}