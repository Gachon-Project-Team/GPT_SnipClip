import styled from 'styled-components';

export const InputContainer = styled.div`
    width: 80%;
    min-height: 300px;
    background-color: white;
    color: black;
    display: flex;
    flex-direction: column;
    border-radius: 15px;
    padding: 20px;
`
export const KeywordInputForm = styled.div`
  display: flex;
  flex-direction: column;  // 자식 요소들을 세로로 정렬
  align-items: center;  // 자식 요소들이 왼쪽에 정렬
  width: 100%;
  padding: 20px;
  background-color: whitesmoke;
  border-radius: 10px;
  min-height: 300px;
  transition: all 0.3s ease;  // 부드러운 전환 효과 추가
  height: ${(props) => (props.isProcessing ? '500px' : 'auto')};  // 진행바가 보이면 높이를 늘림
`;

export const TextElement = styled.h1`
  color: black;
  font-size: 24px;
  margin-bottom: 50px;
`;
export const FormContainer = styled.div`
  display: flex;
  flex-direction: row;
  gap: 5px;
  width: 100%;
`;

export const InputField = styled.input`
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 5px;
  width: 100%;
`;

export const ProgressBarContainer = styled.div`
  width: 100%;
  background-color: #f3f3f3;
  border-radius: 5px;
  margin-top: 20px;
`;

// export const ProgressBar = styled.div`
//   height: 20px;
//   background-color: green;
//   width: ${(props) => props.progress}%;  // 진행 상태에 따라 width를 동적으로 변경
//   border-radius: 5px;
// `;

export const DownloadButton = styled.button`

  margin-top: 10px;
`